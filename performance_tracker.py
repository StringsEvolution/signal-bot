"""
Performance Tracker
Logs every signal with result, generates daily/weekly reports.
"""

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_engine():
    return create_engine(
        os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/signal_bot"),
        pool_pre_ping=True,
    )


# ---------------------------------------------------------------------------
# Log a signal
# ---------------------------------------------------------------------------

def log_signal(signal, features: Optional[dict] = None) -> Optional[int]:
    """
    Insert signal into the signals table. Returns row ID.
    """
    try:
        engine = get_engine()
        sql = text("""
            INSERT INTO signals
                (timestamp, asset, timeframe, direction, entry_price, confidence,
                 expiry_min, reasons, result)
            VALUES
                (:ts, :asset, :tf, :dir, :ep, :conf, :exp, :reasons, NULL)
            RETURNING id
        """)
        with engine.connect() as conn:
            row = conn.execute(sql, {
                "ts":      signal.timestamp,
                "asset":   str(signal.asset),
                "tf":      str(signal.timeframe),
                "dir":     str(signal.direction),
                "ep":      float(signal.entry_price),
                "conf":    float(signal.confidence),
                "exp":     int(signal.expiry_min),
                "reasons": json.dumps(signal.reasons),
            })
            conn.commit()
            return row.fetchone()[0]
    except Exception as exc:
        logger.error(f"Failed to log signal: {exc}")
        return None


def update_result(signal_id: int, result: str):
    """Update signal result: 'win' | 'loss' | 'draw'"""
    try:
        engine = get_engine()
        sql = text("UPDATE signals SET result=:r WHERE id=:id")
        with engine.connect() as conn:
            conn.execute(sql, {"r": result, "id": signal_id})
            conn.commit()
    except Exception as exc:
        logger.error(f"Failed to update result: {exc}")


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def _fetch_signals_for_period(start: datetime, end: datetime) -> pd.DataFrame:
    try:
        engine = get_engine()
        sql = text("""
            SELECT timestamp, asset, timeframe, direction, confidence, result, expiry_min
            FROM signals
            WHERE timestamp BETWEEN :start AND :end
              AND result IS NOT NULL
            ORDER BY timestamp
        """)
        with engine.connect() as conn:
            result = conn.execute(sql, {"start": start, "end": end})
            rows   = result.fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=[
            "timestamp","asset","timeframe","direction","confidence","result","expiry_min"
        ])
    except Exception as exc:
        logger.error(f"Fetch signals failed: {exc}")
        return pd.DataFrame()


def _compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_confidence": 0}

    wins   = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    total  = len(df)

    return {
        "total":          int(total),
        "wins":           int(wins),
        "losses":         int(losses),
        "draws":          int(total - wins - losses),
        "win_rate":       round(wins / total * 100, 2) if total > 0 else 0,
        "avg_confidence": round(df["confidence"].mean(), 1) if not df.empty else 0,
    }


def generate_daily_report(target_date: Optional[date] = None) -> str:
    target = target_date or datetime.utcnow().date()
    start  = datetime.combine(target, datetime.min.time())
    end    = datetime.combine(target, datetime.max.time())

    df   = _fetch_signals_for_period(start, end)
    stats = _compute_stats(df)

    if stats["total"] == 0:
        return f"📊 Daily Report — {target}\n\nNo completed signals today."

    lines = [
        f"📊 *Daily Report — {target}*",
        "",
        f"  Total Signals: {stats['total']}",
        f"  ✅ Wins:       {stats['wins']} ({stats['win_rate']:.1f}%)",
        f"  ❌ Losses:     {stats['losses']}",
        f"  ⚖️ Draws:      {stats['draws']}",
        f"  🤖 Avg Conf:   {stats['avg_confidence']:.1f}%",
        "",
    ]

    # By asset
    if not df.empty:
        lines.append("  *By Asset:*")
        for asset in df["asset"].unique():
            sub   = df[df["asset"] == asset]
            st    = _compute_stats(sub)
            lines.append(f"    {asset}: {st['wins']}W/{st['losses']}L  ({st['win_rate']:.1f}%)")

        lines.append("")
        lines.append("  *By Timeframe:*")
        for tf in df["timeframe"].unique():
            sub = df[df["timeframe"] == tf]
            st  = _compute_stats(sub)
            lines.append(f"    {tf}: {st['wins']}W/{st['losses']}L  ({st['win_rate']:.1f}%)")

    lines.append("")
    lines.append("⚠️ _Trading involves significant risk. Results vary._")
    return "\n".join(lines)


def generate_weekly_report(week_offset: int = 0) -> str:
    today  = datetime.utcnow().date()
    monday = today - timedelta(days=today.weekday()) - timedelta(weeks=week_offset)
    sunday = monday + timedelta(days=6)

    start = datetime.combine(monday, datetime.min.time())
    end   = datetime.combine(sunday, datetime.max.time())

    df    = _fetch_signals_for_period(start, end)
    stats = _compute_stats(df)

    week_label = f"{monday} → {sunday}"

    if stats["total"] == 0:
        return f"📊 Weekly Report ({week_label})\n\nNo completed signals this week."

    lines = [
        f"📊 *Weekly Report*",
        f"*{week_label}*",
        "",
        f"  Total Signals: {stats['total']}",
        f"  ✅ Wins:       {stats['wins']} ({stats['win_rate']:.1f}%)",
        f"  ❌ Losses:     {stats['losses']}",
        f"  ⚖️ Draws:      {stats['draws']}",
        f"  🤖 Avg Conf:   {stats['avg_confidence']:.1f}%",
        "",
    ]

    if not df.empty:
        # Daily breakdown
        lines.append("  *Daily Breakdown:*")
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        for d in sorted(df["date"].unique()):
            sub = df[df["date"] == d]
            st  = _compute_stats(sub)
            lines.append(f"    {d}: {st['wins']}W/{st['losses']}L  ({st['win_rate']:.1f}%)")

    lines.append("")
    lines.append("⚠️ _Trading involves significant risk._")
    return "\n".join(lines)

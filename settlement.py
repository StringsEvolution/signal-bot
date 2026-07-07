"""
Settlement worker (BUG B fix)
-----------------------------
Closes the loop on live signals so the bot actually measures its own accuracy.

Problem it solves:
  Signals were generated and sent, but `performance_tracker.log_signal()` /
  `update_result()` were never called anywhere in the live path. Every signal
  vanished after delivery, so `result` stayed NULL forever and the daily/weekly
  reports always said "No completed signals". There was no win-rate feedback by
  timeframe or session.

What it does:
  1. log_pending(signal, signal_id) — record a fired signal for later scoring.
     (main.py calls performance_tracker.log_signal() to persist the row; this
      module keeps the lightweight in-memory queue of what still needs a result.)
  2. A background worker wakes every SETTLE_TICK_SECS, and for any signal whose
     expiry has elapsed, reads the SETTLED price from the OHLC store and writes
     win/loss/draw back via performance_tracker.update_result().

Settlement price source:
  The streaming engine already persists every closed candle to the OHLC table,
  so the settlement price is arriving there naturally — no extra Deriv/API
  calls, no new failure surface. We read the close of the candle at/after
  (signal.timestamp + expiry_min). This is the price a binary option settles
  against for a next-candle-entry, hold-to-expiry trade.

IMPORTANT — scope of the number this produces:
  This measures the bot's accuracy on the DERIV feed it analyses. Users trade on
  Pocket Option, whose candles do not perfectly align with Deriv's. So this is
  the *engine's* win rate, i.e. "is the signal logic sound", NOT a record of what
  a user actually won on PO. The two track closely during liquid overlap hours
  and can diverge in thin hours. For true PO results, users still log by hand;
  the gap between the two is the execution/feed-divergence cost.
"""

import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List

from data_engine import load_ohlc, TF_MINUTES
import performance_tracker

logger = logging.getLogger(__name__)

# How often the worker checks for signals ready to settle.
SETTLE_TICK_SECS = 15

# Grace period after theoretical expiry before we look for the settle candle,
# to allow the closing candle to actually arrive in the OHLC store via the
# streaming engine (stream + store has a few seconds of lag).
SETTLE_GRACE_SECS = 20

# If a signal still can't be settled this long after expiry (e.g. the market
# closed, stream stalled, or no candle ever arrived), we give up on it so the
# queue doesn't grow without bound. It is recorded as unsettled (left NULL).
SETTLE_ABANDON_SECS = 60 * 60  # 1 hour


@dataclass
class _Pending:
    signal_id:   int
    asset:       str
    timeframe:   str
    direction:   str        # "CALL" | "PUT"
    entry_price: float
    # Naive UTC datetime of the candle CLOSE that triggered the signal
    # (== entry reference). Settlement candle is this + expiry.
    signal_ts:   datetime
    expiry_min:  int
    queued_at:   datetime   # naive UTC, for abandonment timeout


class SettlementWorker:
    """Background thread that scores fired signals once their expiry passes."""

    def __init__(self):
        self._pending: List[_Pending] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- public API ------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="settlement-worker", daemon=True
        )
        self._thread.start()
        logger.info("Settlement worker started.")

    def stop(self):
        self._stop.set()

    def log_pending(self, signal, signal_id: int):
        """
        Register a fired signal (already persisted with a DB id) for later
        scoring. Safe to call from the signal-emission path.
        """
        if signal_id is None:
            return
        try:
            sig_ts = _to_naive_utc(signal.timestamp)
            item = _Pending(
                signal_id   = signal_id,
                asset       = signal.asset,
                timeframe   = signal.timeframe,
                direction   = signal.direction,
                entry_price = float(signal.entry_price),
                signal_ts   = sig_ts,
                expiry_min  = int(signal.expiry_min),
                queued_at   = datetime.utcnow(),
            )
            with self._lock:
                self._pending.append(item)
            logger.info(
                f"⏳ Settlement queued: id={signal_id} {item.asset}/{item.timeframe} "
                f"{item.direction} entry={item.entry_price} "
                f"settle_at={(sig_ts + timedelta(minutes=item.expiry_min)).strftime('%H:%M:%S')} UTC"
            )
        except Exception as exc:
            logger.error(f"Failed to queue settlement for id={signal_id}: {exc}")

    # ---- worker loop -----------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            try:
                self._settle_ready()
            except Exception as exc:
                logger.error(f"Settlement tick error: {exc}", exc_info=True)
            self._stop.wait(SETTLE_TICK_SECS)

    def _settle_ready(self):
        now = datetime.utcnow()
        with self._lock:
            queue = list(self._pending)

        still_pending: List[_Pending] = []
        for item in queue:
            expiry_dt   = item.signal_ts + timedelta(minutes=item.expiry_min)
            ready_after = expiry_dt + timedelta(seconds=SETTLE_GRACE_SECS)

            if now < ready_after:
                still_pending.append(item)          # not due yet
                continue

            result = self._try_settle(item, expiry_dt)
            if result is not None:
                performance_tracker.update_result(item.signal_id, result)
                logger.info(
                    f"✅ Settled id={item.signal_id} {item.asset}/{item.timeframe} "
                    f"{item.direction} → {result.upper()}"
                )
                continue  # done, drop from queue

            # Couldn't settle yet — keep unless we've waited too long.
            if (now - item.queued_at).total_seconds() > SETTLE_ABANDON_SECS:
                logger.warning(
                    f"⚠️ Abandoning settlement id={item.signal_id} "
                    f"{item.asset}/{item.timeframe} — no settle candle after "
                    f"{SETTLE_ABANDON_SECS//60} min (market closed / stream gap). "
                    f"Result left unsettled."
                )
                continue
            still_pending.append(item)

        with self._lock:
            # Preserve any items queued while we were working
            new_items = self._pending[len(queue):]
            self._pending = still_pending + new_items

    def _try_settle(self, item: _Pending, expiry_dt: datetime) -> Optional[str]:
        """
        Return 'win' | 'loss' | 'draw', or None if the settle candle isn't
        available yet.

        Settle candle = the candle whose CLOSE time is at/after expiry_dt.
        OHLC rows are keyed by candle OPEN time, so the candle that OPENS at
        (expiry_dt - one interval) closes at expiry_dt. Equivalently, we want
        the first stored candle with open_time >= (expiry_dt - interval), and
        we take its close.
        """
        try:
            df = load_ohlc(item.asset, item.timeframe, limit=50)
        except Exception as exc:
            logger.error(f"Settle load_ohlc failed id={item.signal_id}: {exc}")
            return None

        if df is None or df.empty:
            return None

        interval = timedelta(minutes=TF_MINUTES.get(item.timeframe, 1))
        target_open = expiry_dt - interval   # open time of the settle candle

        # df timestamps are naive UTC candle OPEN times, ascending.
        ts = df["timestamp"]
        # Find the first candle whose open >= target_open (within a small tol).
        tol = timedelta(seconds=2)
        mask = ts >= (target_open - tol)
        if not mask.any():
            # Newest stored candle is still older than the settle candle —
            # it hasn't arrived yet.
            return None

        settle_row  = df[mask].iloc[0]
        settle_close = float(settle_row["close"])

        # Sanity: the matched candle should be the settle candle, not something
        # far in the future. If the nearest candle is more than one interval
        # past target, data has a gap — accept it but note it.
        settle_open = settle_row["timestamp"]
        if isinstance(settle_open, str):
            settle_open = datetime.fromisoformat(settle_open)
        settle_open = _to_naive_utc(settle_open)

        entry = item.entry_price
        if settle_close > entry:
            price_dir = "CALL"   # price rose
        elif settle_close < entry:
            price_dir = "PUT"    # price fell
        else:
            return "draw"

        return "win" if price_dir == item.direction else "loss"


def _to_naive_utc(dt) -> datetime:
    """Coerce any datetime/Timestamp to naive UTC."""
    if dt is None:
        return datetime.utcnow()
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    # pandas Timestamp has tzinfo attr too
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt if isinstance(dt, datetime) else datetime.utcnow()


# Module-level singleton so main.py and the emission path share one worker.
worker = SettlementWorker()

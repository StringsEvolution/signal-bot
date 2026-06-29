"""
Data Engine — fetches OHLC candlestick data and stores in PostgreSQL.
Primary source: Deriv API (WebSocket, real-time, unlimited, Nigeria supported)
Fallback: Twelve Data (REST, 800 req/day)
Last resort: Synthetic data
"""

import os
import time
import json
import logging
import asyncio
import threading
import websockets
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ASSETS     = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]
TIMEFRAMES = ["M1", "M2", "M3", "M5", "M15"]

TF_MINUTES = {"M1": 1,  "M2": 2,  "M3": 3,  "M5": 5,  "M15": 15}
TF_EXPIRY  = {"M1": 1,  "M2": 2,  "M3": 3,  "M5": 5,  "M15": 15}

DERIV_SYMBOLS = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "XAUUSD": "frxXAUUSD",
    "USDJPY": "frxUSDJPY",
    "BTCUSD": "cryBTCUSD",
}

DERIV_GRANULARITY = {"M1": 60, "M2": 120, "M3": 180, "M5": 300, "M15": 900}

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=36544"

_db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_engine():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/signal_bot"
    )
    return create_engine(db_url, pool_pre_ping=True)


def init_db():
    engine = get_engine()
    ddl = """
    CREATE TABLE IF NOT EXISTS ohlc_data (
        id          SERIAL PRIMARY KEY,
        asset       VARCHAR(10) NOT NULL,
        timeframe   VARCHAR(5)  NOT NULL,
        timestamp   TIMESTAMPTZ NOT NULL,
        open        NUMERIC(18,6) NOT NULL,
        high        NUMERIC(18,6) NOT NULL,
        low         NUMERIC(18,6) NOT NULL,
        close       NUMERIC(18,6) NOT NULL,
        volume      NUMERIC(18,2) DEFAULT 0,
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (asset, timeframe, timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_ohlc_asset_tf_ts
        ON ohlc_data (asset, timeframe, timestamp DESC);

    CREATE TABLE IF NOT EXISTS signals (
        id          SERIAL PRIMARY KEY,
        timestamp   TIMESTAMPTZ NOT NULL,
        asset       VARCHAR(10) NOT NULL,
        timeframe   VARCHAR(5)  NOT NULL,
        direction   VARCHAR(4)  NOT NULL,
        entry_price NUMERIC(18,6),
        confidence  NUMERIC(5,2),
        expiry_min  INT,
        reasons     TEXT,
        result      VARCHAR(4),
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS performance_log (
        id          SERIAL PRIMARY KEY,
        date        DATE NOT NULL,
        asset       VARCHAR(10),
        timeframe   VARCHAR(5),
        total       INT DEFAULT 0,
        wins        INT DEFAULT 0,
        losses      INT DEFAULT 0,
        win_rate    NUMERIC(5,2),
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Database initialised.")


# ---------------------------------------------------------------------------
# Deriv API — Primary real-time data source
# ---------------------------------------------------------------------------

async def _fetch_deriv_async(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    symbol      = DERIV_SYMBOLS.get(asset)
    granularity = DERIV_GRANULARITY.get(timeframe)

    if not symbol or not granularity:
        raise ValueError(f"Unsupported asset/timeframe: {asset}/{timeframe}")

    token = os.getenv("DERIV_API_TOKEN", "").strip()

    async with websockets.connect(DERIV_WS_URL, ping_interval=20) as ws:

        if token:
            await ws.send(json.dumps({"authorize": token}))
            auth_resp = json.loads(await ws.recv())
            if auth_resp.get("error"):
                logger.debug(
                    f"Deriv auth skipped ({auth_resp['error']['message']}) "
                    f"— fetching unauthenticated."
                )

        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": n_candles,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
        }
        await ws.send(json.dumps(request))
        response = json.loads(await ws.recv())

        if response.get("error"):
            raise ValueError(f"Deriv error: {response['error']['message']}")

        candles = response.get("candles", [])
        if not candles:
            raise ValueError(f"No candles returned from Deriv for {asset}/{timeframe}")

        rows = []
        for c in candles:
            rows.append({
                "timestamp": pd.to_datetime(c["epoch"], unit="s"),
                "open":      float(c["open"]),
                "high":      float(c["high"]),
                "low":       float(c["low"]),
                "close":     float(c["close"]),
                "volume":    0.0,
            })

        df = pd.DataFrame(rows)
        # Safe sort — handles duplicates and NaN timestamps
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"✅ Deriv: fetched {len(df)} candles for {asset}/{timeframe} (real-time)")
        return df


def _fetch_deriv(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_fetch_deriv_async(asset, timeframe, n_candles))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Twelve Data — Fallback with key rotation
# ---------------------------------------------------------------------------

class TwelveDataKeyManager:
    def __init__(self):
        self.keys        = self._load_keys()
        self.current_idx = 0
        self.exhausted   = set()

    def _load_keys(self):
        keys = []
        for i in range(1, 11):
            k = os.getenv(f"TWELVE_DATA_KEY_{i}", "").strip()
            if k:
                keys.append(k)
        if not keys:
            k = os.getenv("TWELVE_DATA_KEY", "").strip()
            if k:
                keys.append(k)
        if keys:
            logger.info(f"TwelveDataKeyManager: {len(keys)} key(s) loaded (fallback).")
        return keys

    @property
    def active_key(self) -> str:
        return self.keys[self.current_idx] if self.keys else ""

    @property
    def has_keys(self) -> bool:
        return bool(self.keys) and len(self.exhausted) < len(self.keys)

    def rotate(self, reason: str = "rate limit"):
        self.exhausted.add(self.current_idx)
        available = [i for i in range(len(self.keys)) if i not in self.exhausted]
        if not available:
            logger.error("All Twelve Data keys exhausted.")
            return
        self.current_idx = available[0]
        logger.warning(f"Twelve Data key rotated ({reason}). On key #{self.current_idx + 1}.")

    def reset_daily(self):
        self.exhausted   = set()
        self.current_idx = 0
        logger.info("TwelveDataKeyManager: daily reset.")

    def is_rate_limit(self, data: dict, status_code: int) -> bool:
        if status_code == 429:
            return True
        if data.get("status") == "error":
            msg = data.get("message", "").lower()
            if any(x in msg for x in ["api credits", "rate limit", "too many",
                                       "exceeded", "limit reached", "upgrade"]):
                return True
        return False


_key_manager = TwelveDataKeyManager()

def get_key_manager() -> TwelveDataKeyManager:
    return _key_manager


def _fetch_twelve_data(asset: str, timeframe: str) -> pd.DataFrame:
    TD_INTERVAL = {"M1": "1min", "M5": "5min", "M15": "15min"}
    interval    = TD_INTERVAL.get(timeframe)

    if not interval:
        raise ValueError(f"Twelve Data does not support timeframe {timeframe}")

    if asset == "XAUUSD":
        symbol = "XAU/USD"
    elif asset == "BTCUSD":
        symbol = "BTC/USD"
    else:
        symbol = f"{asset[:3]}/{asset[3:]}"

    km          = get_key_manager()
    max_retries = len(km.keys) if km.keys else 1

    for attempt in range(max_retries):
        if not km.has_keys:
            raise ValueError("All Twelve Data keys exhausted.")

        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}&interval={interval}"
            f"&outputsize=200&apikey={km.active_key}"
        )

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                km.rotate("HTTP 429")
                continue

            resp.raise_for_status()
            data = resp.json()

            if km.is_rate_limit(data, resp.status_code):
                km.rotate(data.get("message", "limit"))
                continue

            if data.get("status") == "error":
                raise ValueError(f"Twelve Data error: {data.get('message')}")

            if "values" not in data or not data["values"]:
                raise ValueError(f"No data from Twelve Data for {asset}")

            rows = []
            for v in data["values"]:
                rows.append({
                    "timestamp": pd.to_datetime(v["datetime"]),
                    "open":   float(v["open"]),
                    "high":   float(v["high"]),
                    "low":    float(v["low"]),
                    "close":  float(v["close"]),
                    "volume": float(v.get("volume", 0)),
                })

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"⚠️ Twelve Data fallback: {len(df)} candles for {asset}/{timeframe}")
            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            logger.warning(f"Twelve Data network error: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)

    raise ValueError(f"All Twelve Data fetch attempts failed for {asset}/{timeframe}")


# ---------------------------------------------------------------------------
# Synthetic data — Last resort
# ---------------------------------------------------------------------------

def _synthetic_ohlc(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    seed = abs(hash(asset + timeframe)) % (2**31)
    rng  = np.random.default_rng(seed)

    base_prices = {
        "EURUSD": 1.0850, "GBPUSD": 1.2700, "XAUUSD": 2350.0,
        "USDJPY": 150.0,  "BTCUSD": 65000.0,
    }
    sigma = {
        "EURUSD": 0.0003, "GBPUSD": 0.0004, "XAUUSD": 0.8,
        "USDJPY": 0.05,   "BTCUSD": 500.0,
    }
    base    = base_prices.get(asset, 1.0)
    sig     = sigma.get(asset, 0.0003)
    minutes = TF_MINUTES[timeframe]
    now     = datetime.utcnow().replace(second=0, microsecond=0)
    start   = now - timedelta(minutes=minutes * n_candles)
    timestamps = [start + timedelta(minutes=i * minutes) for i in range(n_candles)]
    closes     = base + np.cumsum(rng.normal(0, sig, n_candles))

    rows = []
    for i, (ts, close) in enumerate(zip(timestamps, closes)):
        open_  = closes[i - 1] if i > 0 else close
        body   = abs(open_ - close)
        high   = max(open_, close) + abs(rng.normal(0, max(body * 1.2, sig * 0.8)))
        low    = min(open_, close) - abs(rng.normal(0, max(body * 1.2, sig * 0.8)))
        rows.append({"timestamp": ts, "open": open_, "high": high,
                     "low": low, "close": close, "volume": rng.integers(200, 1000)})

    logger.warning(f"⚠️ Using SYNTHETIC data for {asset}/{timeframe} — not for production!")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Primary fetch — Deriv → Twelve Data → Synthetic
# ---------------------------------------------------------------------------

def fetch_ohlc(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    # 1. Try Deriv
    try:
        df = _fetch_deriv(asset, timeframe, n_candles)
        return df.tail(n_candles).reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Deriv fetch failed for {asset}/{timeframe}: {exc}")

    # 2. Try Twelve Data
    km = get_key_manager()
    if km.has_keys:
        try:
            df = _fetch_twelve_data(asset, timeframe)
            return df.tail(n_candles).reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Twelve Data fallback failed: {exc}")

    # 3. Synthetic last resort
    return _synthetic_ohlc(asset, timeframe, n_candles)


# ---------------------------------------------------------------------------
# Store / Load — thread-safe via _db_lock
# ---------------------------------------------------------------------------

def store_ohlc(asset: str, timeframe: str, df: pd.DataFrame):
    if df.empty:
        return

    rows = [
        {
            "asset":     asset,
            "timeframe": timeframe,
            "timestamp": row["timestamp"],
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row.get("volume", 0)),
        }
        for _, row in df.iterrows()
    ]

    upsert = text("""
        INSERT INTO ohlc_data (asset, timeframe, timestamp, open, high, low, close, volume)
        VALUES (:asset, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (asset, timeframe, timestamp) DO UPDATE
            SET open=EXCLUDED.open, high=EXCLUDED.high,
                low=EXCLUDED.low,  close=EXCLUDED.close,
                volume=EXCLUDED.volume
    """)

    with _db_lock:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(upsert, rows)
            conn.commit()

    logger.debug(f"Stored {len(df)} rows for {asset}/{timeframe}.")


def load_ohlc(asset: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """
    Load most recent candles from PostgreSQL.
    Thread-safe via _db_lock.
    Includes safety guards against NaN/duplicate timestamps that cause segfault.
    """
    sql = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlc_data
        WHERE asset=:asset AND timeframe=:tf
        ORDER BY timestamp DESC
        LIMIT :lim
    """)

    with _db_lock:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql, {"asset": asset, "tf": timeframe, "lim": limit})
            rows   = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    records = [
        {
            "timestamp": row[0],
            "open":   float(row[1]),
            "high":   float(row[2]),
            "low":    float(row[3]),
            "close":  float(row[4]),
            "volume": float(row[5]) if row[5] is not None else 0.0,
        }
        for row in rows
    ]

    df = pd.DataFrame(records)

    # Convert timestamps — strip timezone info to avoid Python 3.13 + pandas segfault
    # on sort_values with timezone-aware TIMESTAMPTZ from PostgreSQL
    def _safe_ts(ts):
        try:
            t = pd.to_datetime(ts, errors="coerce", utc=True)
            if pd.isna(t):
                return pd.NaT
            return t.tz_localize(None) if t.tzinfo is None else t.tz_convert(None)
        except Exception:
            return pd.NaT

    df["timestamp"] = df["timestamp"].apply(_safe_ts)

    # Safety guards
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"])

    if len(df) < 5:
        logger.warning(f"Insufficient data for {asset}/{timeframe}: only {len(df)} rows")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Use numpy argsort instead of pandas sort_values to avoid segfault
    idx = df["timestamp"].values.argsort()
    df  = df.iloc[idx].reset_index(drop=True)
    return df


def refresh_all():
    """Fetch all assets/timeframes in parallel."""
    import concurrent.futures

    pairs      = [(asset, tf) for asset in ASSETS for tf in TIMEFRAMES]
    start_time = time.time()

    def fetch_and_store(asset: str, tf: str):
        try:
            df = fetch_ohlc(asset, tf)
            store_ohlc(asset, tf, df)
            logger.info(f"✅ Refreshed {asset}/{tf}")
        except Exception as exc:
            logger.error(f"refresh_all failed for {asset}/{tf}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(fetch_and_store, asset, tf) for asset, tf in pairs]
        concurrent.futures.wait(futures)

    elapsed = time.time() - start_time
    logger.info(f"⚡ All {len(pairs)} pairs refreshed in {elapsed:.1f}s (parallel)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    refresh_all()
    print("Data refresh complete.")

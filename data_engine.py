"""
Data Engine — fetches OHLC candlestick data and stores in PostgreSQL.
Supports EURUSD, GBPUSD, XAUUSD, USDJPY, BTCUSD across M1, M5, M15 timeframes.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ASSETS = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]
TIMEFRAMES = ["M1", "M5", "M15"]

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}
TF_EXPIRY  = {"M1": 3,  "M5": 5, "M15": 15}

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
    """Create tables if they do not exist."""
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
# Rotating API Key Manager
# ---------------------------------------------------------------------------

class TwelveDataKeyManager:
    """
    Rotates between two (or more) Twelve Data API keys.
    Switches automatically when a key hits its rate or daily limit.

    Set in Railway environment variables:
        TWELVE_DATA_KEY_1=your_first_key
        TWELVE_DATA_KEY_2=your_second_key

    Legacy single-key fallback still supported:
        TWELVE_DATA_KEY=your_key
    """

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
        # fallback: single key
        if not keys:
            k = os.getenv("TWELVE_DATA_KEY", "").strip()
            if k:
                keys.append(k)
        if keys:
            logger.info(f"TwelveDataKeyManager: {len(keys)} API key(s) loaded.")
        else:
            logger.warning("TwelveDataKeyManager: no API keys found — will use synthetic data.")
        return keys

    @property
    def active_key(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self.current_idx]

    @property
    def has_keys(self) -> bool:
        return bool(self.keys) and len(self.exhausted) < len(self.keys)

    def rotate(self, reason: str = "rate limit"):
        """Mark current key as exhausted and switch to the next available one."""
        self.exhausted.add(self.current_idx)
        available = [i for i in range(len(self.keys)) if i not in self.exhausted]

        if not available:
            logger.error(
                f"All {len(self.keys)} Twelve Data key(s) exhausted ({reason}). "
                f"Falling back to synthetic data until midnight UTC reset."
            )
            return

        self.current_idx = available[0]
        logger.warning(
            f"Twelve Data key rotated ({reason}). "
            f"Now on key #{self.current_idx + 1}. "
            f"{len(available) - 1} backup key(s) still available."
        )

    def reset_daily(self):
        """Call at midnight UTC — Twelve Data resets quotas daily."""
        self.exhausted   = set()
        self.current_idx = 0
        logger.info("TwelveDataKeyManager: daily reset — all keys active again.")

    def is_rate_limit(self, data: dict, status_code: int) -> bool:
        """Detect rate limit from HTTP status or API response body."""
        if status_code == 429:
            return True
        if data.get("status") == "error":
            msg = data.get("message", "").lower()
            if any(x in msg for x in ["api credits", "rate limit", "too many",
                                       "exceeded", "limit reached", "upgrade"]):
                return True
        return False


# Singleton — shared across all fetch calls in this process
_key_manager = TwelveDataKeyManager()

def get_key_manager() -> TwelveDataKeyManager:
    return _key_manager


# ---------------------------------------------------------------------------
# OHLC fetch — Twelve Data with key rotation, synthetic fallback
# ---------------------------------------------------------------------------

def _fetch_twelve_data(asset: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch from Twelve Data. Rotates to backup key on rate limit.
    """
    TD_INTERVAL = {"M1": "1min", "M5": "5min", "M15": "15min"}
    interval = TD_INTERVAL[timeframe]

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
            raise ValueError("All Twelve Data API keys exhausted.")

        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}"
            f"&interval={interval}"
            f"&outputsize=200"
            f"&apikey={km.active_key}"
        )

        try:
            resp = requests.get(url, timeout=15)

            # HTTP 429 — rotate immediately and retry
            if resp.status_code == 429:
                logger.warning(f"Key #{km.current_idx + 1} HTTP 429. Rotating...")
                km.rotate("HTTP 429")
                continue

            resp.raise_for_status()
            data = resp.json()

            # API-level rate limit error — rotate and retry
            if km.is_rate_limit(data, resp.status_code):
                logger.warning(
                    f"Key #{km.current_idx + 1} limit: {data.get('message', '')}. Rotating..."
                )
                km.rotate(data.get("message", "limit"))
                continue

            if data.get("status") == "error":
                raise ValueError(f"Twelve Data error: {data.get('message', 'unknown')}")

            if "values" not in data or not data["values"]:
                raise ValueError(f"No data returned from Twelve Data for {asset}")

            rows = []
            for vals in data["values"]:
                rows.append({
                    "timestamp": pd.to_datetime(vals["datetime"]),
                    "open":      float(vals["open"]),
                    "high":      float(vals["high"]),
                    "low":       float(vals["low"]),
                    "close":     float(vals["close"]),
                    "volume":    float(vals.get("volume", 0)),
                })

            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            logger.info(
                f"Fetched {len(df)} candles for {asset} {timeframe} "
                f"from Twelve Data (key #{km.current_idx + 1})."
            )
            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            logger.warning(f"Network error for {asset}/{timeframe}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue

    raise ValueError(f"All fetch attempts failed for {asset}/{timeframe}")


def _synthetic_ohlc(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    """
    Deterministic synthetic OHLC for offline/demo use.
    Seeds with asset+timeframe so results are reproducible.
    """
    seed = abs(hash(asset + timeframe)) % (2**31)
    rng  = np.random.default_rng(seed)

    base_prices = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2700,
        "XAUUSD": 2350.0,
        "USDJPY": 150.0,
        "BTCUSD": 65000.0,
    }
    base = base_prices.get(asset, 1.0)
    sigma = {
        "EURUSD": 0.0003,
        "GBPUSD": 0.0004,
        "XAUUSD": 0.8,
        "USDJPY": 0.05,
        "BTCUSD": 500.0,
    }.get(asset, 0.0003)

    minutes = TF_MINUTES[timeframe]
    now     = datetime.utcnow().replace(second=0, microsecond=0)
    start   = now - timedelta(minutes=minutes * n_candles)

    timestamps = [start + timedelta(minutes=i * minutes) for i in range(n_candles)]
    closes     = base + np.cumsum(rng.normal(0, sigma, n_candles))

    rows = []
    for i, (ts, close) in enumerate(zip(timestamps, closes)):
        open_      = closes[i - 1] if i > 0 else close
        body       = abs(open_ - close)
        upper_wick = abs(rng.normal(0, max(body * 1.2, sigma * 0.8)))
        lower_wick = abs(rng.normal(0, max(body * 1.2, sigma * 0.8)))
        high       = max(open_, close) + upper_wick
        low        = min(open_, close) - lower_wick
        volume     = rng.integers(200, 1000)
        rows.append({"timestamp": ts, "open": open_, "high": high,
                     "low": low, "close": close, "volume": volume})

    return pd.DataFrame(rows)


def fetch_ohlc(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    """
    Primary fetch function.
    Tries Twelve Data (with key rotation) first, falls back to synthetic data.
    """
    km = get_key_manager()
    if km.has_keys:
        try:
            df = _fetch_twelve_data(asset, timeframe)
            return df.tail(n_candles).reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Twelve Data fetch failed ({exc}), using synthetic data.")

    df = _synthetic_ohlc(asset, timeframe, n_candles)
    logger.info(f"Using synthetic data for {asset} {timeframe} ({len(df)} candles).")
    return df


def store_ohlc(asset: str, timeframe: str, df: pd.DataFrame):
    """Upsert OHLC rows into PostgreSQL."""
    engine = get_engine()
    upsert = text("""
        INSERT INTO ohlc_data (asset, timeframe, timestamp, open, high, low, close, volume)
        VALUES (:asset, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (asset, timeframe, timestamp) DO UPDATE
            SET open=EXCLUDED.open, high=EXCLUDED.high,
                low=EXCLUDED.low,  close=EXCLUDED.close,
                volume=EXCLUDED.volume
    """)
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(upsert, {
                "asset": asset, "timeframe": timeframe,
                "timestamp": row["timestamp"],
                "open": float(row["open"]),   "high": float(row["high"]),
                "low":  float(row["low"]),    "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        conn.commit()
    logger.debug(f"Stored {len(df)} rows for {asset} {timeframe}.")


def load_ohlc(asset: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """Load the most recent candles from PostgreSQL."""
    engine = get_engine()
    sql = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlc_data
        WHERE asset=:asset AND timeframe=:tf
        ORDER BY timestamp DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"asset": asset, "tf": timeframe, "lim": limit})
        rows   = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def refresh_all():
    """Fetch and store fresh OHLC data for all assets and timeframes."""
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlc(asset, tf)
                store_ohlc(asset, tf, df)
            except Exception as exc:
                logger.error(f"refresh_all failed for {asset}/{tf}: {exc}")
            time.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    refresh_all()
    print("Data refresh complete.")

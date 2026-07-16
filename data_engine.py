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
from sqlalchemy import create_engine, text, event
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ASSETS     = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
TIMEFRAMES = ["M1", "M2", "M3", "M5", "M15"]

TF_MINUTES = {"M1": 1,  "M2": 2,  "M3": 3,  "M5": 5,  "M15": 15}
TF_EXPIRY  = {"M1": 1,  "M2": 2,  "M3": 3,  "M5": 5,  "M15": 15}

DERIV_SYMBOLS = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "USDCHF": "frxUSDCHF",
}

DERIV_GRANULARITY = {"M1": 60, "M2": 120, "M3": 180, "M5": 300, "M15": 900}

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=36544"


# ---------------------------------------------------------------------------
# Forming-candle registry
# ---------------------------------------------------------------------------
# The live stream publishes each in-progress (still-open) candle here so the
# pre-alert worker can peek at it a few seconds before it closes. Keyed by
# (asset, timeframe) -> snapshot dict {open_time, open, high, low, close}.
import threading as _threading
_forming_lock = _threading.Lock()
_forming_candles: dict = {}


def _publish_forming(asset: str, timeframe: str, snapshot: dict):
    with _forming_lock:
        _forming_candles[(asset, timeframe)] = dict(snapshot)


def get_forming_candle(asset: str, timeframe: str) -> Optional[dict]:
    """Return the latest in-progress candle snapshot, or None."""
    with _forming_lock:
        snap = _forming_candles.get((asset, timeframe))
        return dict(snap) if snap else None


# ---------------------------------------------------------------------------
# Helper — force clean numeric/datetime dtypes (prevents segfault)
# ---------------------------------------------------------------------------

def _clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        try:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return df


def _sort_by_timestamp_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df

    ts = np.asarray(df["timestamp"].values, dtype="datetime64[ns]")
    order = np.argsort(ts, kind="stable")

    def col(name, default=0.0):
        if name in df.columns:
            return np.asarray(df[name].values, dtype="float64")[order]
        return np.full(len(order), default, dtype="float64")

    return pd.DataFrame({
        "timestamp": ts[order],
        "open":   col("open"),
        "high":   col("high"),
        "low":    col("low"),
        "close":  col("close"),
        "volume": col("volume"),
    })


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/signal_bot"
        )
        # pool_recycle raised 300s -> 1800s: fewer forced reconnects means
        # fewer TLS handshakes, which on managed Postgres (Neon) count toward
        # the data-transfer quota. pool_pre_ping stays on for resilience but
        # can be disabled with DB_PRE_PING=0 if you need to shave every byte.
        pre_ping = os.getenv("DB_PRE_PING", "1") != "0"
        _engine = create_engine(
            db_url,
            pool_pre_ping=pre_ping,
            pool_size=5,
            max_overflow=5,
            pool_timeout=10,
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
            connect_args={
                "connect_timeout": 10,
            },
        )

        @event.listens_for(_engine, "connect")
        def _set_session_timeout(dbapi_conn, conn_record):
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("SET statement_timeout = 15000")
            finally:
                cursor.close()

        return _engine


class DBUnavailable(Exception):
    """Raised when the database cannot be reached after retries.

    This is deliberately distinct from a transient/programming error so the
    caller (main.py) can decide to back off *inside the same process* instead
    of letting the process die and be restart-stormed by the platform. When
    the DB is down for a hard reason (e.g. Neon/Postgres transfer-quota
    exceeded), a tight restart loop hammers the provider and burns even more
    quota — the opposite of what we want.
    """


def wait_for_db(max_attempts: int = 6, base_delay: float = 2.0,
                max_delay: float = 60.0) -> bool:
    """Try to open a DB connection, retrying with exponential backoff.

    Returns True on success. Raises DBUnavailable if all attempts fail.
    Quota-exhaustion errors are detected and surfaced clearly so the operator
    knows the fix is 'upgrade / reduce transfer', not 'restart the bot'.
    """
    engine = get_engine()
    delay = base_delay
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            if attempt > 1:
                logger.info("Database reachable after %d attempt(s).", attempt)
            return True
        except Exception as exc:  # noqa: BLE001 - we re-classify below
            last_err = exc
            msg = str(exc).lower()
            is_quota = ("data transfer quota" in msg or "exceeded" in msg
                        and "quota" in msg)
            # Dispose so we don't keep a poisoned/half-open pool around.
            try:
                engine.dispose()
            except Exception:
                pass
            if is_quota:
                logger.error(
                    "DB rejected connection due to a PROVIDER QUOTA limit "
                    "(attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
            else:
                logger.warning(
                    "DB not ready (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
            if attempt < max_attempts:
                logger.info("Retrying DB connection in %.0fs...", delay)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
    raise DBUnavailable(
        f"Database unreachable after {max_attempts} attempts. "
        f"Last error: {last_err}"
    )


def init_db():
    engine = get_engine()
    # Gate: don't run DDL until the DB actually answers. Raises DBUnavailable
    # (caught in main) instead of letting a raw OperationalError kill the
    # process and trigger a platform restart storm.
    wait_for_db()
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
# Deriv API — Parallel batch fetch (PRIMARY for seeding/backfill)
# ---------------------------------------------------------------------------

async def _fetch_all_deriv_async(pairs: list, n_candles: int = 200) -> dict:
    """
    ONE WebSocket connection, ALL pairs fetched IN PARALLEL via req_id matching.
    Sends all requests at once then collects responses concurrently.
    Reduces 25-pair fetch from ~320s (sequential) → ~8s (parallel).
    """
    token   = os.getenv("DERIV_API_TOKEN", "").strip()
    results = {}

    async with websockets.connect(DERIV_WS_URL, open_timeout=15, ping_interval=20) as ws:

        # Authenticate once if token available
        if token:
            await ws.send(json.dumps({"authorize": token}))
            auth_resp = json.loads(await ws.recv())
            if auth_resp.get("error"):
                logger.debug("Deriv auth skipped — fetching unauthenticated.")

        # Build req_id → (asset, timeframe) map and request list
        req_map          = {}
        req_id           = 1
        requests_to_send = []

        for asset, timeframe in pairs:
            symbol      = DERIV_SYMBOLS.get(asset)
            granularity = DERIV_GRANULARITY.get(timeframe)
            if not symbol or not granularity:
                continue
            req_map[req_id] = (asset, timeframe)
            requests_to_send.append({
                "ticks_history":     symbol,
                "adjust_start_time": 1,
                "count":             n_candles,
                "end":               "latest",
                "granularity":       granularity,
                "style":             "candles",
                "req_id":            req_id,
            })
            req_id += 1

        # Fire ALL requests at once — don't wait for individual responses
        for req in requests_to_send:
            await ws.send(json.dumps(req))
        logger.info(f"Deriv: sent {len(requests_to_send)} requests in parallel")

        # Collect responses as they arrive — matched by req_id (order not guaranteed)
        expected = len(requests_to_send)
        received = 0
        deadline = asyncio.get_event_loop().time() + 30  # 30s hard timeout

        while received < expected:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(f"Deriv batch timeout — got {received}/{expected} responses")
                break
            try:
                raw      = await asyncio.wait_for(ws.recv(), timeout=remaining)
                response = json.loads(raw)
            except asyncio.TimeoutError:
                logger.warning(f"Deriv recv timeout — got {received}/{expected}")
                break
            except Exception as exc:
                logger.warning(f"Deriv recv error: {exc}")
                break

            # Skip auth/heartbeat responses not in our map
            r_id = response.get("req_id")
            if r_id not in req_map:
                continue

            received += 1
            asset, timeframe = req_map[r_id]

            if response.get("error"):
                logger.warning(f"Deriv error {asset}/{timeframe}: {response['error']['message']}")
                continue

            candles = response.get("candles", [])
            if not candles:
                logger.warning(f"No candles from Deriv for {asset}/{timeframe}")
                continue

            rows = [
                {
                    "timestamp": pd.to_datetime(c["epoch"], unit="s"),
                    "open":  float(c["open"]),
                    "high":  float(c["high"]),
                    "low":   float(c["low"]),
                    "close": float(c["close"]),
                    "volume": 0.0,
                }
                for c in candles
            ]
            df = pd.DataFrame(rows)
            df = _clean_dtypes(df)
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            df = df.drop_duplicates(subset=["timestamp"])
            df = _sort_by_timestamp_safe(df)
            results[(asset, timeframe)] = df
            logger.info(f"✅ Deriv: {asset}/{timeframe} — {len(df)} candles")

    logger.info(f"Deriv parallel fetch complete: {len(results)}/{len(pairs)} pairs succeeded")
    return results


# ---------------------------------------------------------------------------
# Real-time streaming engine — push-based, drives live signals
# ---------------------------------------------------------------------------

async def _stream_asset(asset: str, timeframes: list, on_close, stop_event: asyncio.Event):
    symbol  = DERIV_SYMBOLS.get(asset)
    if not symbol:
        return
    req_map = {i + 1: tf for i, tf in enumerate(timeframes)}

    while not stop_event.is_set():
        last_candle = {}
        try:
            async with websockets.connect(
                DERIV_WS_URL, open_timeout=15, ping_interval=20, ping_timeout=20
            ) as ws:
                for req_id, tf in req_map.items():
                    await ws.send(json.dumps({
                        "ticks_history":     symbol,
                        "style":             "candles",
                        "granularity":       DERIV_GRANULARITY[tf],
                        "subscribe":         1,
                        "count":             1,
                        "end":               "latest",
                        "req_id":            req_id,
                    }))
                logger.info(f"📡 Live stream connected: {asset} ({len(req_map)} timeframes)")

                market_closed_count = 0

                while not stop_event.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=40)
                    msg = json.loads(raw)

                    if msg.get("error"):
                        err_msg = msg['error'].get('message', '')
                        logger.warning(f"Stream error {asset}: {err_msg}")
                        # If market is closed, back off instead of hammering reconnects.
                        # Count errors across all timeframe subscriptions (5 per asset).
                        # Once all 5 confirm closed, sleep until the market reopens.
                        if "presently closed" in err_msg.lower() or "market is closed" in err_msg.lower():
                            market_closed_count += 1
                            if market_closed_count >= len(req_map):
                                # All timeframes confirmed closed — sleep 10 minutes
                                logger.info(
                                    f"💤 {asset} market closed — pausing stream for 10 minutes "
                                    f"(will auto-reconnect when market reopens)"
                                )
                                await asyncio.sleep(600)
                                market_closed_count = 0
                                break  # break inner loop to trigger reconnect
                        continue

                    market_closed_count = 0  # reset on any successful message

                    tf = req_map.get(msg.get("req_id"))
                    if not tf:
                        continue

                    mtype = msg.get("msg_type")

                    if mtype == "candles":
                        candles = msg.get("candles") or []
                        if candles:
                            c = candles[-1]
                            last_candle[tf] = {
                                "open_time": int(c["epoch"]),
                                "open":  float(c["open"]),
                                "high":  float(c["high"]),
                                "low":   float(c["low"]),
                                "close": float(c["close"]),
                            }

                    elif mtype == "ohlc":
                        c = msg["ohlc"]
                        snapshot = {
                            "open_time": int(c["open_time"]),
                            "open":  float(c["open"]),
                            "high":  float(c["high"]),
                            "low":   float(c["low"]),
                            "close": float(c["close"]),
                        }
                        # Publish the in-progress candle so the pre-alert worker
                        # can peek at it a few seconds before it closes.
                        _publish_forming(asset, tf, snapshot)
                        prev = last_candle.get(tf)
                        if prev is not None and prev["open_time"] != snapshot["open_time"]:
                            closed = prev
                            asyncio.create_task(on_close(asset, tf, closed))
                        last_candle[tf] = snapshot

        except asyncio.TimeoutError:
            logger.warning(f"{asset} stream silent for 40s — reconnecting.")
        except Exception as exc:
            logger.warning(f"{asset} stream dropped ({exc}) — reconnecting in 3s.")
            await asyncio.sleep(3)


async def run_streaming_engine(on_close, stop_event: Optional[asyncio.Event] = None):
    stop_event = stop_event or asyncio.Event()
    await asyncio.gather(*[
        _stream_asset(asset, TIMEFRAMES, on_close, stop_event)
        for asset in ASSETS
    ])


async def _fetch_deriv_async(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    result = await _fetch_all_deriv_async([(asset, timeframe)], n_candles)
    df = result.get((asset, timeframe))
    if df is None or df.empty:
        raise ValueError(f"No data from Deriv for {asset}/{timeframe}")
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
            df = _clean_dtypes(df)
            df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            df = df.drop_duplicates(subset=["timestamp"])
            df = _sort_by_timestamp_safe(df)
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
        "USDJPY": 150.0,  "BTCUSD": 65000.0, "USDCHF": 0.8800,
    }
    sigma = {
        "EURUSD": 0.0003, "GBPUSD": 0.0004, "XAUUSD": 0.8,
        "USDJPY": 0.05,   "BTCUSD": 500.0,  "USDCHF": 0.0003,
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
    return _clean_dtypes(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Primary fetch — Deriv → Twelve Data → Synthetic
# ---------------------------------------------------------------------------

def fetch_ohlc(asset: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    try:
        df = _fetch_deriv(asset, timeframe, n_candles)
        return df.tail(n_candles).reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Deriv fetch failed for {asset}/{timeframe}: {exc}")

    km = get_key_manager()
    if km.has_keys:
        try:
            df = _fetch_twelve_data(asset, timeframe)
            return df.tail(n_candles).reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Twelve Data fallback failed: {exc}")

    return _synthetic_ohlc(asset, timeframe, n_candles)


# ---------------------------------------------------------------------------
# Store / Load
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

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(upsert, rows)
        conn.commit()

    logger.debug(f"Stored {len(df)} rows for {asset}/{timeframe}.")


def load_ohlc(asset: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    sql = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlc_data
        WHERE asset=:asset AND timeframe=:tf
        ORDER BY timestamp DESC
        LIMIT :lim
    """)

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(sql, {"asset": asset, "tf": timeframe, "lim": limit})
        rows   = result.fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = list(reversed(rows))

    seen = set()
    ts_list, op_list, hi_list, lo_list, cl_list, vol_list = [], [], [], [], [], []
    for r in rows:
        ts = r[0]
        if ts is None or ts in seen:
            continue
        seen.add(ts)
        try:
            o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            v = float(r[5]) if r[5] is not None else 0.0
        except (TypeError, ValueError):
            continue
        ts_list.append(pd.Timestamp(ts))
        op_list.append(o); hi_list.append(h); lo_list.append(l)
        cl_list.append(c); vol_list.append(v)

    if len(ts_list) < 5:
        logger.warning(f"Insufficient data for {asset}/{timeframe}: only {len(ts_list)} rows")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ts_arr = pd.to_datetime(ts_list, utc=True).tz_localize(None)

    df = pd.DataFrame({
        "timestamp": np.asarray(ts_arr, dtype="datetime64[ns]"),
        "open":   np.asarray(op_list,  dtype="float64"),
        "high":   np.asarray(hi_list,  dtype="float64"),
        "low":    np.asarray(lo_list,  dtype="float64"),
        "close":  np.asarray(cl_list,  dtype="float64"),
        "volume": np.asarray(vol_list, dtype="float64"),
    })
    return df


def refresh_all():
    """
    Fetch all assets/timeframes in ONE Deriv WebSocket connection (PARALLEL).
    25 pairs fetched simultaneously → completes in ~8s instead of ~320s.
    Falls back per-pair to Twelve Data for any that fail.
    """
    pairs      = [(asset, tf) for asset in ASSETS for tf in TIMEFRAMES]
    start_time = time.time()

    # Parallel batch fetch over a single Deriv WS connection
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        deriv_results = loop.run_until_complete(_fetch_all_deriv_async(pairs))
        loop.close()
    except Exception as exc:
        logger.warning(f"Deriv batch fetch failed: {exc} — falling back to Twelve Data")
        deriv_results = {}

    import concurrent.futures

    def finish_pair(asset: str, tf: str):
        df = deriv_results.get((asset, tf))
        if df is not None and not df.empty:
            store_ohlc(asset, tf, df)
            return
        km = get_key_manager()
        if km.has_keys:
            try:
                df = _fetch_twelve_data(asset, tf)
                store_ohlc(asset, tf, df)
                logger.info(f"✅ Twelve Data fallback: {asset}/{tf}")
                return
            except Exception as exc:
                logger.warning(f"Twelve Data also failed {asset}/{tf}: {exc}")
        try:
            df = _synthetic_ohlc(asset, tf)
            store_ohlc(asset, tf, df)
            logger.warning(f"⚠️ Using synthetic data for {asset}/{tf}")
        except Exception as exc:
            logger.error(f"All sources failed for {asset}/{tf}: {exc}")

    missed = [p for p in pairs if p not in deriv_results]
    if missed:
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(finish_pair, a, tf) for a, tf in pairs]
            concurrent.futures.wait(futures)
    else:
        for asset, tf in pairs:
            finish_pair(asset, tf)

    elapsed = time.time() - start_time
    logger.info(f"⚡ All {len(pairs)} pairs refreshed in {elapsed:.1f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    refresh_all()
    print("Data refresh complete.")

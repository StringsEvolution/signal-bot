"""
Pocket Option Engine — per-user OTC price streaming for Pocket Option.

Design goals (see requirements this file implements):
  1. Uses the `pocket-option` PyPI SDK (pure async, event-driven,
     `pip install pocket-option`). Import is lazy + guarded so the rest of
     the bot (Deriv signals, Telegram, etc.) keeps working unmodified even
     if this package isn't installed or a user's environment can't run it
     (the SDK currently requires Python 3.13+).
  2. Fully per-user: each Telegram user supplies their own PO_SESSION /
     PO_UID (collected via the Telegram conversation flow in
     telegram_bot.py and stored via UserManager). There is no shared
     global Pocket Option account.
  3. If a user has no/invalid credentials, ONLY that user's stream is
     skipped — never crashes the bot or blocks other users/Deriv signals.
  4. Each user's ticks are aggregated into OHLC candles per timeframe and
     written through `data_engine.store_ohlc`, i.e. the EXACT same table
     and schema the existing Deriv pipeline already uses. This means
     `signal_engine.generate_signal()` can be called on Pocket Option
     assets with zero changes — the data just "looks like" any other
     asset/timeframe pair. Pocket Option OTC assets are stored under an
     `_otc` suffixed asset code (e.g. "EURUSD_otc") so they never collide
     with the Deriv rows for "EURUSD".
"""

import os
import time
import logging
import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PLATFORM = "pocket_option"

# Same timeframe set as the rest of the bot (data_engine.TF_MINUTES), kept
# as a local constant too so this module has no hard import-time dependency
# on data_engine internals beyond store_ohlc.
TF_MINUTES = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M15": 15}

# Default OTC watchlist if a user hasn't set their own PO_ASSETS.
# NOTE: Pocket Option's symbol names can differ from these (e.g. gold may be
# listed as "Gold OTC"). _po_asset_code() normalizes whatever the broker sends,
# but if an asset never streams, check the exact symbol name in your PO account
# and set PO_ASSETS accordingly.
DEFAULT_PO_ASSETS = [
    a.strip() for a in os.getenv(
        "PO_ASSETS",
        "EURUSD_otc,GBPUSD_otc,XAUUSD_otc,EURJPY_otc,AUDCAD_otc,USDJPY_otc"
    ).split(",") if a.strip()
]

CandleCallback = Optional[Callable[[str, str, dict], None]]


# ---------------------------------------------------------------------------
# Engine heartbeat — lets an external watchdog (main.py) confirm the Pocket
# Option thread is still alive and cycling. Purely observational: nothing in
# the OTC or Deriv signal path depends on it. Thread-safe via a lock because
# it's written by the engine thread and read by the watchdog thread.
# ---------------------------------------------------------------------------

_health_lock = threading.Lock()
_health = {
    "started":      False,   # True once the SDK check passed and the loop began
    "last_beat":    None,    # epoch seconds of the most recent cycle
    "cycles":       0,       # number of rescan cycles completed
    "live_streams": 0,       # streams currently supervised
}


def _beat(live_streams: int):
    with _health_lock:
        _health["started"]      = True
        _health["last_beat"]    = time.time()
        _health["cycles"]      += 1
        _health["live_streams"] = live_streams


def get_engine_health() -> dict:
    """
    Snapshot of the engine's liveness for the watchdog / health checks.

    Returns keys:
      started       — has the engine thread actually begun cycling?
      last_beat     — epoch seconds of the last cycle (None if never)
      age_seconds   — seconds since last_beat (None if never)
      cycles        — completed rescan cycles
      live_streams  — streams currently supervised
    """
    with _health_lock:
        snap = dict(_health)
    snap["age_seconds"] = (
        None if snap["last_beat"] is None else round(time.time() - snap["last_beat"], 1)
    )
    return snap


# ---------------------------------------------------------------------------
# Per-asset tick -> multi-timeframe candle aggregator
# ---------------------------------------------------------------------------

class _CandleAggregator:
    """Buckets raw price ticks into OHLC candles for every timeframe at once."""

    def __init__(self):
        # {timeframe: {"open_time": int, "open": f, "high": f, "low": f, "close": f}}
        self._forming: dict = {}

    def on_tick(self, price: float, ts_epoch: int) -> list:
        """Feed one tick. Returns a list of (timeframe, closed_candle) tuples
        for any timeframe whose bucket just rolled over."""
        closed = []
        for tf, minutes in TF_MINUTES.items():
            bucket_secs = minutes * 60
            open_time = (ts_epoch // bucket_secs) * bucket_secs
            cur = self._forming.get(tf)

            if cur is None or cur["open_time"] != open_time:
                # Bucket rolled over — emit the previous one if it exists.
                if cur is not None:
                    closed.append((tf, dict(cur)))
                self._forming[tf] = {
                    "open_time": open_time,
                    "open": price, "high": price, "low": price, "close": price,
                }
            else:
                cur["high"] = max(cur["high"], price)
                cur["low"]  = min(cur["low"], price)
                cur["close"] = price

        return closed


# ---------------------------------------------------------------------------
# Helpers to normalize whatever shape the SDK hands back for a price update
# ---------------------------------------------------------------------------

def _extract_asset_and_price(item) -> Optional[tuple]:
    """
    The `update_close_value` event can hand back a pydantic model, a dict,
    or (per some SDK versions) a list of either. This defensively pulls
    (asset_symbol, price, epoch_seconds) out of one item, or returns None
    if the shape is unrecognised (in which case we just skip that tick —
    never crash the stream over it).
    """
    try:
        get = (lambda k: getattr(item, k, None)) if not isinstance(item, dict) \
              else (lambda k: item.get(k))

        asset = get("asset") or get("symbol") or get("active") or get("name")
        price = get("value") or get("close") or get("price") or get("rate")
        ts    = get("time") or get("timestamp") or get("ts")

        if asset is None or price is None:
            return None

        if ts is None:
            ts_epoch = int(time.time())
        elif isinstance(ts, (int, float)):
            ts_epoch = int(ts)
        elif isinstance(ts, datetime):
            ts_epoch = int(ts.timestamp())
        else:
            ts_epoch = int(time.time())

        return str(asset), float(price), ts_epoch
    except Exception:
        return None


def _po_asset_code(raw_asset: str) -> str:
    """Normalize a Pocket Option asset name to our `_otc` storage code,
    kept within the existing VARCHAR(10) `asset` column.

    Standard 6-char FX pairs fit exactly: EURUSD -> EURUSD_otc (10 chars).
    For longer symbols (e.g. BITCOIN) we truncate the BASE and always keep
    the `_otc` suffix intact — truncating the whole string would strip the
    suffix (BITCOIN_ot) and break asset matching + display.
    """
    MAX_LEN = 10
    SUFFIX  = "_otc"

    code = raw_asset.upper().replace("/", "").replace("-", "").replace(" ", "")
    # Strip any existing OTC marker to get the clean base symbol.
    if code.endswith("OTC"):
        code = code[:-3]
    code = code.replace("_OTC", "").rstrip("_")

    base = code[: MAX_LEN - len(SUFFIX)]   # leave room for "_otc"
    return base + SUFFIX


# ---------------------------------------------------------------------------
# One user's Pocket Option stream
# ---------------------------------------------------------------------------

async def _run_user_stream(telegram_id: str, credentials: dict, is_demo: bool,
                            assets: list, on_candle: CandleCallback):
    """
    Connects ONE user's Pocket Option session, subscribes to their asset
    list, aggregates ticks into candles, and stores them via
    data_engine.store_ohlc. Any failure here (bad session, network error,
    SDK exception) is caught and logged — it only takes this one user's
    stream down, never the bot.
    """
    session = credentials.get("session") or credentials.get("PO_SESSION")
    uid     = credentials.get("uid") or credentials.get("PO_UID")

    if not session or not uid:
        logger.warning(
            f"[pocket_option] user={telegram_id}: missing session/uid — "
            f"skipping this user only."
        )
        return "skip"

    try:
        from pocket_option import PocketOptionClient
        from pocket_option.models import AuthorizationData
    except ImportError:
        logger.warning(
            "[pocket_option] `pocket-option` package not installed "
            "(pip install pocket-option, requires Python 3.13+) — "
            "Pocket Option integration disabled."
        )
        return "skip"

    watch_assets = assets or DEFAULT_PO_ASSETS
    aggregators: dict = {a: _CandleAggregator() for a in watch_assets}

    from data_engine import store_ohlc  # local import avoids any import cycle

    client = PocketOptionClient()

    # Set by the auth-failure handler so the supervisor can tell an expired
    # session (don't retry blindly) apart from a transient network drop.
    auth_failed = {"flag": False}

    @client.on.connect
    async def _on_connect(_data):
        # Auth is passed directly into client.connect(auth=...) for this SDK
        # version, so nothing is emitted here — this is purely a log hook.
        logger.info(f"[pocket_option] user={telegram_id}: socket connected.")

    # If the SDK exposes an auth-failure event, use it to flag expired
    # sessions. Wrapped in try/except because event names vary by SDK version;
    # a missing event simply means we fall back to backoff-based retries.
    try:
        @client.on.auth_error
        async def _on_auth_error(_data):
            auth_failed["flag"] = True
            logger.warning(f"[pocket_option] user={telegram_id}: authentication rejected (session likely expired).")
    except Exception:
        pass

    @client.on.success_auth
    async def _on_success_auth(_data):
        logger.info(
            f"[pocket_option] user={telegram_id}: authorized "
            f"({'DEMO' if is_demo else 'REAL'}) — subscribing {len(watch_assets)} assets"
        )

    @client.on.update_close_value
    async def _on_update_close_value(data):
        items = data if isinstance(data, (list, tuple)) else [data]
        for item in items:
            parsed = _extract_asset_and_price(item)
            if not parsed:
                continue
            raw_asset, price, ts_epoch = parsed
            code = _po_asset_code(raw_asset)
            if code not in aggregators:
                # Not one of this user's watched assets — ignore.
                continue

            for tf, candle in aggregators[code].on_tick(price, ts_epoch):
                try:
                    ts = pd.to_datetime(candle["open_time"], unit="s")
                    df = pd.DataFrame([{
                        "timestamp": ts,
                        "open": candle["open"], "high": candle["high"],
                        "low": candle["low"], "close": candle["close"],
                        "volume": 0.0,
                    }])
                    store_ohlc(code, tf, df)
                    if on_candle:
                        on_candle(code, tf, candle)
                except Exception as exc:
                    logger.error(
                        f"[pocket_option] user={telegram_id}: store candle "
                        f"failed for {code}/{tf} — {exc}"
                    )

    # ---- Connect ----------------------------------------------------------
    # The SDK's real signature is:
    #   connect(url, headers=None, auth=None, wait=True, wait_timeout=1, retry=False)
    # so the auth payload is passed straight into connect() (NOT emitted after
    # the connect event), and browser-like headers are required — Pocket Option
    # rejects plain server clients that don't send an Origin/User-Agent.
    #
    # PO_WS_URL is env-driven because Pocket Option runs several regional
    # endpoints, and DEMO vs REAL are different hosts. Copy the exact URL from
    # DevTools → Network → Socket → (the socket showing "updateStream" price
    # frames) → Headers → Request URL, and set it in Railway.
    PO_WS_URL = os.getenv(
        "PO_WS_URL",
        "wss://demo-api-eu.po.market/socket.io/?EIO=4&transport=websocket"
    )

    po_headers = {
        "Origin": os.getenv("PO_ORIGIN", "https://pocketoption.com"),
        "User-Agent": os.getenv(
            "PO_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
    }

    try:
        auth_data = AuthorizationData.model_validate({
            "session": session,
            "isDemo": 1 if is_demo else 0,
            "uid": int(uid),
            "platform": 2,
            "isFastHistory": True,
            "isOptimized": True,
        })
    except Exception as exc:
        logger.error(
            f"[pocket_option] user={telegram_id}: invalid credentials payload — {exc}"
        )
        return "auth_failed"

    try:
        await client.connect(
            PO_WS_URL,
            headers=po_headers,
            auth=auth_data,
            wait=True,
            wait_timeout=float(os.getenv("PO_CONNECT_TIMEOUT", "20")),
        )
        # connect() returns once the socket is up; wait() blocks until the
        # stream closes, keeping this user's task alive while ticks flow in.
        await client.wait()
    except Exception as exc:
        logger.error(f"[pocket_option] user={telegram_id}: stream error — {exc}")
        if auth_failed["flag"]:
            return "auth_failed"
        return "error"
    finally:
        # Always tear the client down. Without this the aiohttp session and
        # TCP connector leak on every reconnect ("Unclosed client session"),
        # which slowly exhausts the container over many retries.
        for closer in ("shutdown", "disconnect"):
            fn = getattr(client, closer, None)
            if fn is None:
                continue
            try:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    # Clean exit (socket closed without exception). Treat expired-auth as such.
    return "auth_failed" if auth_failed["flag"] else "closed"


async def _supervise_user_stream(telegram_id, credentials, is_demo, assets,
                                 on_candle, stop_event: "asyncio.Event"):
    """
    Production supervisor for ONE user's Pocket Option stream.

    Wraps _run_user_stream with:
      * automatic reconnect on transient drops,
      * exponential backoff (5s → 300s cap) so a flaky/expired session never
        hammers Pocket Option,
      * a circuit breaker: after MAX_AUTH_FAILS consecutive auth rejections
        the user's credentials are deactivated and they're DM'd to reconnect,
        so an expired session stops retrying forever.
    A single user going down never affects any other user or the Deriv path.
    """
    BASE_DELAY   = 5
    MAX_DELAY    = 300
    MAX_AUTH_FAILS = 3

    delay        = BASE_DELAY
    auth_fails   = 0

    while not stop_event.is_set():
        outcome = await _run_user_stream(
            telegram_id=telegram_id, credentials=credentials,
            is_demo=is_demo, assets=assets, on_candle=on_candle,
        )

        if outcome == "skip":
            # Missing creds — nothing to retry.
            return

        if outcome == "auth_failed":
            auth_fails += 1
            logger.warning(
                f"[pocket_option] user={telegram_id}: auth failure "
                f"{auth_fails}/{MAX_AUTH_FAILS}."
            )
            if auth_fails >= MAX_AUTH_FAILS:
                # Circuit breaker: stop retrying an expired session and tell
                # the user to reconnect.
                try:
                    from user_manager import UserManager
                    UserManager().deactivate_platform_credentials(
                        telegram_id, PLATFORM, reason="repeated auth failure (expired session)"
                    )
                except Exception as exc:
                    logger.error(f"[pocket_option] user={telegram_id}: deactivate failed — {exc}")
                try:
                    from telegram_bot import _send_message
                    _send_message(
                        str(telegram_id),
                        "⚠️ <b>Pocket Option disconnected</b>\n\n"
                        "Your Pocket Option session looks expired, so OTC signals "
                        "are paused for your account.\n\n"
                        "Please run /connectpo to reconnect with a fresh session."
                    )
                except Exception as exc:
                    logger.error(f"[pocket_option] user={telegram_id}: expiry DM failed — {exc}")
                return
        else:
            # Successful connection cycle (or transient network error) —
            # reset the auth-failure counter and backoff.
            auth_fails = 0
            delay = BASE_DELAY

        if stop_event.is_set():
            return

        logger.info(f"[pocket_option] user={telegram_id}: reconnecting in {delay}s...")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
        delay = min(delay * 2, MAX_DELAY)


async def run_pocket_option_streams(on_candle: CandleCallback = None,
                                    running: dict = None,
                                    stop_event: "asyncio.Event" = None):
    """
    Entry point: looks up every user with ACTIVE Pocket Option credentials
    and runs their stream (via the resilient supervisor) concurrently. One
    bad user never stops the rest. Safe to call even if no users have
    connected yet (just logs and returns).

    `running` is a registry of telegram_id -> asyncio.Task for streams that
    are already live, so the periodic rescan only STARTS streams for newly
    connected users instead of spawning duplicates for existing ones.
    """
    try:
        from user_manager import UserManager
    except Exception as exc:
        logger.error(f"[pocket_option] could not import UserManager — {exc}")
        return

    if running is None:
        running = {}
    if stop_event is None:
        stop_event = asyncio.Event()

    users = UserManager().get_all_platform_users(PLATFORM)

    # Reap finished tasks (auth-deactivated / permanently skipped users) so a
    # later reconnect via /connectpo can start a fresh stream for them.
    for tid in [t for t, task in running.items() if task.done()]:
        running.pop(tid, None)

    if not users:
        if not running:
            logger.info("[pocket_option] no users have connected a Pocket Option account yet.")
        return

    started = 0
    for u in users:
        tid = str(u["telegram_id"])
        if tid in running and not running[tid].done():
            continue  # already streaming — don't double-spawn
        assets = [_po_asset_code(a) if not a.endswith("_otc") else a for a in u["assets"]] or []
        task = asyncio.ensure_future(
            _supervise_user_stream(
                telegram_id=tid,
                credentials=u["credentials"],
                is_demo=u["is_demo"],
                assets=assets,
                on_candle=on_candle,
                stop_event=stop_event,
            )
        )
        running[tid] = task
        started += 1

    if started:
        logger.info(f"[pocket_option] started {started} new stream(s); {len(running)} total live.")


# ---------------------------------------------------------------------------
# Background-thread launcher (for main.py to call without blocking the
# existing Deriv streaming loop / event loop)
# ---------------------------------------------------------------------------

def start_pocket_option_engine(on_candle: CandleCallback = None,
                                rescan_interval: int = 300):
    """
    Runs the Pocket Option engine on its own thread + event loop so it
    never competes with or blocks the main Deriv streaming loop. Every
    `rescan_interval` seconds it re-reads the user list, so a user who
    connects mid-session (via /connectpo) is picked up on the next pass
    without a bot restart. Entirely optional — if the SDK isn't installed
    or no users are connected, this just idles quietly.
    """
    def _thread_main():
        try:
            import pocket_option  # noqa: F401 — presence check only
        except ImportError:
            logger.info(
                "[pocket_option] SDK not installed — Pocket Option integration "
                "is disabled (this does not affect the rest of the bot)."
            )
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _loop_forever():
            # Persistent across rescans: streams already live + a shared stop
            # event. The rescan then only STARTS newly connected users and
            # never duplicates or restarts healthy streams.
            running: dict = {}
            stop_event = asyncio.Event()
            while True:
                try:
                    await run_pocket_option_streams(
                        on_candle=on_candle, running=running, stop_event=stop_event,
                    )
                except Exception as exc:
                    logger.error(f"[pocket_option] engine cycle error: {exc}", exc_info=True)
                # Heartbeat AFTER the cycle so a stuck rescan shows up as a
                # stale beat to the watchdog. Counts only live (not-done) tasks.
                live = sum(1 for t in running.values() if not t.done())
                _beat(live)
                await asyncio.sleep(rescan_interval)

        try:
            loop.run_until_complete(_loop_forever())
        except Exception as exc:
            logger.error(f"[pocket_option] engine thread crashed: {exc}", exc_info=True)

    t = threading.Thread(target=_thread_main, name="PocketOptionEngine", daemon=True)
    t.start()
    logger.info("[pocket_option] engine thread launched (no-op if SDK/users are absent).")
    return t

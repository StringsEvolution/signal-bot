"""
IQ Option OTC engine.

Mirrors pocket_option_engine.py so it plugs into the SAME signal pipeline:
each user's live OTC candles are normalized and written through
data_engine.store_ohlc (the exact same ohlc_data table), then main.py runs
generate_signal() and delivers via send_otc_signal() — identical to how
Pocket Option and Deriv signals flow.

Key differences from the Pocket Option engine, driven by the `iqoptionapi`
library (https://github.com/iqoptionapi/iqoptionapi):

  * The library is SYNCHRONOUS/threaded, not async. So each user runs on its
    own daemon thread with a blocking poll loop, rather than an asyncio task.
  * Auth is email + PASSWORD (not a session token). Credentials are stored
    per-user exactly like Pocket Option, under platform="iq_option", and are
    encrypted at rest by user_manager via crypto_util. A leaked password is a
    full account takeover, so this engine never logs credential values.
  * The library returns full realtime CANDLES (open/max/min/close), so there
    is no tick aggregation to do — we read closed candles directly.
  * OTC assets are named "EURUSD-OTC" (hyphen, uppercase), vs Pocket Option's
    "EURUSD_otc". We normalize to the same internal "_otc" storage codes so
    the signal logic and per-user asset matching stay consistent across both
    brokers.

Safety / isolation guarantees (same as the PO engine):
  * A missing library, bad credentials, or a single user's failure NEVER
    crashes the bot or affects Deriv. Everything is wrapped and that user is
    simply skipped / retried with backoff.
  * If the `iqoptionapi` package isn't installed, the engine disables itself
    cleanly with a log line and does nothing else.
"""

import os
import time
import logging
import threading
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PLATFORM = "iq_option"

# Same timeframe set as the rest of the bot. IQ's stream `size` is the candle
# duration in SECONDS, so we map each timeframe to its second-count here.
TF_MINUTES = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M15": 15}
TF_SECONDS = {tf: m * 60 for tf, m in TF_MINUTES.items()}

# IQ Option's candle API only accepts a FIXED set of durations (in seconds).
# 180s (M3) is NOT one of them, so calling start_candles_stream / 
# get_realtime_candles with 180 makes the library log
#   **error** get_realtime_candles() please input right "size"
# on EVERY poll for EVERY asset — the once-per-second log flood. We filter to
# supported sizes at stream-start below, so M3 is simply skipped on IQ's native
# feed. (M3 still streams fine on the Deriv pipeline / VIP channel; it's only
# IQ Option that can't provide a native 3-minute candle.)
IQ_VALID_SIZES = {
    1, 5, 10, 15, 30, 60, 120, 300, 600, 900,
    1800, 3600, 7200, 14400, 28800, 43200, 86400, 604800, 2592000,
}

# Default OTC watchlist if a user hasn't set their own IQ_ASSETS. IQ uses
# hyphenated uppercase OTC names; _iq_asset_code() converts to internal codes.
DEFAULT_IQ_ASSETS = [
    a.strip() for a in os.getenv(
        "IQ_ASSETS",
        "EURUSD-OTC,GBPUSD-OTC,USDJPY-OTC,EURJPY-OTC,AUDCAD-OTC"
    ).split(",") if a.strip()
]

CandleCallback = Optional[Callable[[str, str, dict], None]]

# How often (seconds) to rescan the DB for newly connected IQ users.
RESCAN_INTERVAL = int(os.getenv("IQ_RESCAN_INTERVAL", "300"))

# How many candles of history to keep per stream in the library's buffer.
MAXDICT = int(os.getenv("IQ_MAXDICT", "50"))


# ---------------------------------------------------------------------------
# Engine heartbeat (same contract as pocket_option_engine.get_engine_health)
# ---------------------------------------------------------------------------

_health_lock = threading.Lock()
_health = {"started": False, "last_beat": None, "cycles": 0, "live_streams": 0}


def _beat(live_streams: int):
    with _health_lock:
        _health["started"]      = True
        _health["last_beat"]    = time.time()
        _health["cycles"]      += 1
        _health["live_streams"] = live_streams


def get_engine_health() -> dict:
    with _health_lock:
        snap = dict(_health)
    snap["age_seconds"] = (
        None if snap["last_beat"] is None else round(time.time() - snap["last_beat"], 1)
    )
    return snap


# ---------------------------------------------------------------------------
# Asset code normalization
# ---------------------------------------------------------------------------

def _iq_asset_code(raw_asset: str) -> str:
    """Normalize an IQ OTC name to our internal `_otc` storage code, capped to
    the VARCHAR(10) `asset` column — matching the Pocket Option convention so
    both brokers share the same downstream signal/matching logic.

    'EURUSD-OTC' -> 'EURUSD_otc'.  Longer bases are truncated but ALWAYS keep
    the '_otc' suffix (never 'EURUSD_ot')."""
    MAX_LEN = 10
    SUFFIX  = "_otc"

    code = raw_asset.upper().replace("/", "").replace(" ", "")
    # Strip any OTC marker (hyphen or underscore form) to get the clean base.
    code = code.replace("-OTC", "").replace("_OTC", "").rstrip("-_")

    base = code[: MAX_LEN - len(SUFFIX)]
    return base + SUFFIX


def _iq_symbol(raw_asset: str) -> str:
    """Convert a configured asset into the exact symbol IQ's API expects,
    e.g. 'EURUSD_otc' or 'eurusd-otc' -> 'EURUSD-OTC'."""
    base = raw_asset.upper().replace("/", "").replace(" ", "")
    base = base.replace("-OTC", "").replace("_OTC", "").rstrip("-_")
    return f"{base}-OTC"


# ---------------------------------------------------------------------------
# Per-user stream (runs on its own thread; the IQ library is synchronous)
# ---------------------------------------------------------------------------

def _run_user_stream(telegram_id, credentials: dict, is_demo: bool,
                     assets: list, on_candle: CandleCallback,
                     stop_event: threading.Event) -> str:
    """
    Blocking stream for ONE user. Returns an outcome string the supervisor
    acts on: "skip" (no retry), "auth_failed" (bad creds), "error" (retry),
    or "stopped".

    Any failure is contained to this user — it never propagates to the bot,
    Deriv, or other IQ users.
    """
    email    = (credentials or {}).get("email")
    password = (credentials or {}).get("password")

    if not email or not password:
        logger.warning(
            f"[iq_option] user={telegram_id}: missing email/password — "
            f"skipping this user only."
        )
        return "skip"

    try:
        from iqoptionapi.stable_api import IQ_Option
    except ImportError:
        logger.warning(
            "[iq_option] `iqoptionapi` package not installed — IQ Option "
            "integration disabled. (pip install "
            "https://github.com/iqoptionapi/iqoptionapi/archive/refs/heads/master.zip)"
        )
        return "skip"

    from data_engine import store_ohlc
    import pandas as pd
    from datetime import datetime, timezone

    watch = assets or DEFAULT_IQ_ASSETS
    symbols = []
    for a in watch:
        try:
            symbols.append(_iq_symbol(a))
        except Exception:
            continue
    symbols = list(dict.fromkeys(symbols))  # dedupe, keep order

    Iq = IQ_Option(email, password)

    try:
        check, reason = Iq.connect()
    except Exception as exc:
        logger.error(f"[iq_option] user={telegram_id}: connect() raised — {exc}")
        return "error"

    if not check:
        # Distinguish bad credentials (don't retry forever) from transient
        # issues. IQ returns a JSON reason string on auth failure.
        reason_str = str(reason).lower()
        if "invalid" in reason_str or "credential" in reason_str or "password" in reason_str:
            logger.warning(
                f"[iq_option] user={telegram_id}: authentication rejected "
                f"(bad credentials or 2FA required)."
            )
            return "auth_failed"
        logger.error(f"[iq_option] user={telegram_id}: connect failed — {reason}")
        return "error"

    logger.info(
        f"[iq_option] user={telegram_id}: authorized "
        f"({'DEMO' if is_demo else 'REAL'}) — subscribing {len(symbols)} assets"
    )

    # Select demo vs real balance if the library supports it.
    try:
        Iq.change_balance("PRACTICE" if is_demo else "REAL")
    except Exception as exc:
        logger.warning(f"[iq_option] user={telegram_id}: change_balance failed — {exc}")

    # Start one realtime candle stream per (symbol, timeframe).
    started = []
    for sym in symbols:
        for tf, secs in TF_SECONDS.items():
            if secs not in IQ_VALID_SIZES:
                # e.g. M3 (180s). Logged ONCE per (sym, tf) here instead of
                # flooding the log on every realtime poll. Skipping means this
                # (sym, tf) is never added to `started`, so the poll loop below
                # never requests it.
                logger.info(
                    f"[iq_option] user={telegram_id}: skipping {sym}/{tf} "
                    f"({secs}s) — not an IQ-supported candle size."
                )
                continue
            try:
                Iq.start_candles_stream(sym, secs, MAXDICT)
                started.append((sym, tf, secs))
            except Exception as exc:
                logger.warning(
                    f"[iq_option] user={telegram_id}: could not start "
                    f"{sym}/{tf} stream — {exc}"
                )

    if not started:
        logger.error(f"[iq_option] user={telegram_id}: no streams started.")
        try:
            Iq.close_connect()
        except Exception:
            pass
        return "error"

    # Track the last CLOSED candle open-time we emitted per (code, tf), so we
    # only fire a signal once, when a candle finishes.
    last_emitted: dict = {}

    try:
        while not stop_event.is_set():
            if not _iq_is_connected(Iq):
                logger.warning(f"[iq_option] user={telegram_id}: connection dropped.")
                return "error"

            now_epoch = int(time.time())

            for sym, tf, secs in started:
                try:
                    candles = Iq.get_realtime_candles(sym, secs)
                except Exception:
                    continue
                if not candles:
                    continue

                code = _iq_asset_code(sym)

                # candles is {open_time: {open,max,min,close,volume,...}}.
                # A candle is CLOSED once its open_time + secs <= now.
                for open_time in sorted(candles.keys()):
                    c = candles[open_time]
                    try:
                        ot = int(open_time)
                    except (TypeError, ValueError):
                        continue

                    if ot + secs > now_epoch:
                        continue  # still forming
                    key = (code, tf)
                    if last_emitted.get(key) == ot:
                        continue  # already emitted this candle
                    # Only emit if it's newer than the last one we sent.
                    if key in last_emitted and ot <= last_emitted[key]:
                        continue

                    try:
                        close_dt = datetime.fromtimestamp(ot + secs, tz=timezone.utc)
                        row = {
                            "timestamp": close_dt,
                            "open":  float(c.get("open")),
                            "high":  float(c.get("max")),
                            "low":   float(c.get("min")),
                            "close": float(c.get("close")),
                            "volume": float(c.get("volume", 0) or 0),
                        }
                    except (TypeError, ValueError):
                        continue

                    # Persist to the SAME table Deriv/PO use.
                    try:
                        store_ohlc(code, tf, pd.DataFrame([row]))
                    except Exception as exc:
                        logger.error(
                            f"[iq_option] user={telegram_id}: store_ohlc "
                            f"{code}/{tf} failed — {exc}"
                        )
                        continue

                    last_emitted[key] = ot

                    if on_candle is not None:
                        candle = {
                            "open_time": ot,
                            "open":  row["open"], "high": row["high"],
                            "low":   row["low"],  "close": row["close"],
                        }
                        try:
                            on_candle(code, tf, candle)
                        except Exception as exc:
                            logger.error(
                                f"[iq_option] user={telegram_id}: on_candle "
                                f"{code}/{tf} failed — {exc}"
                            )

            # Poll interval: IQ streams push into the buffer continuously; we
            # sweep once a second for freshly-closed candles.
            stop_event.wait(timeout=1.0)

        return "stopped"

    except Exception as exc:
        logger.error(f"[iq_option] user={telegram_id}: stream loop error — {exc}")
        return "error"
    finally:
        for sym, tf, secs in started:
            try:
                Iq.stop_candles_stream(sym, secs)
            except Exception:
                pass
        try:
            Iq.close_connect()
        except Exception:
            pass


def _iq_is_connected(Iq) -> bool:
    """Best-effort connection check across library versions."""
    for name in ("check_connect", "is_connected"):
        fn = getattr(Iq, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return True  # don't kill the stream over a check error
    return True


# ---------------------------------------------------------------------------
# Supervisor: reconnect + backoff + auth circuit breaker (per user)
# ---------------------------------------------------------------------------

def _supervise_user_stream(telegram_id, credentials, is_demo, assets,
                           on_candle, stop_event: threading.Event):
    BASE_DELAY     = 5
    MAX_DELAY      = 300
    MAX_AUTH_FAILS = 3

    delay      = BASE_DELAY
    auth_fails = 0

    while not stop_event.is_set():
        outcome = _run_user_stream(
            telegram_id=telegram_id, credentials=credentials,
            is_demo=is_demo, assets=assets, on_candle=on_candle,
            stop_event=stop_event,
        )

        if outcome in ("skip", "stopped"):
            return

        if outcome == "auth_failed":
            auth_fails += 1
            logger.warning(
                f"[iq_option] user={telegram_id}: auth failure "
                f"{auth_fails}/{MAX_AUTH_FAILS}."
            )
            if auth_fails >= MAX_AUTH_FAILS:
                try:
                    from user_manager import UserManager
                    UserManager().deactivate_platform_credentials(
                        telegram_id, PLATFORM,
                        reason="repeated auth failure (bad credentials / 2FA)",
                    )
                except Exception as exc:
                    logger.error(f"[iq_option] user={telegram_id}: deactivate failed — {exc}")
                try:
                    from telegram_bot import _send_message
                    _send_message(
                        str(telegram_id),
                        "⚠️ <b>IQ Option disconnected</b>\n\n"
                        "Your IQ Option login was rejected (wrong password, or "
                        "2FA is enabled). OTC signals from IQ Option are paused "
                        "for your account.\n\n"
                        "Turn OFF 2FA on IQ Option, then run /connectiq to "
                        "reconnect."
                    )
                except Exception as exc:
                    logger.error(f"[iq_option] user={telegram_id}: notify failed — {exc}")
                return
        else:
            auth_fails = 0
            delay = BASE_DELAY

        if stop_event.is_set():
            return
        logger.info(f"[iq_option] user={telegram_id}: reconnecting in {delay}s...")
        stop_event.wait(timeout=delay)
        delay = min(delay * 2, MAX_DELAY)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _rescan_and_launch(on_candle, running: dict, stop_event: threading.Event):
    try:
        from user_manager import UserManager
    except Exception as exc:
        logger.error(f"[iq_option] could not import UserManager — {exc}")
        return

    users = UserManager().get_all_platform_users(PLATFORM)

    # Reap finished threads so a later /connectiq can restart them.
    for tid in [t for t, th in running.items() if not th.is_alive()]:
        running.pop(tid, None)

    if not users:
        if not running:
            logger.info("[iq_option] no users have connected an IQ Option account yet.")
        _beat(len(running))
        return

    started = 0
    for u in users:
        tid = str(u["telegram_id"])
        if tid in running and running[tid].is_alive():
            continue
        assets = u["assets"] or []
        th = threading.Thread(
            target=_supervise_user_stream,
            args=(tid, u["credentials"], u["is_demo"], assets, on_candle, stop_event),
            name=f"IQStream-{tid}",
            daemon=True,
        )
        th.start()
        running[tid] = th
        started += 1

    if started:
        logger.info(f"[iq_option] started {started} new stream(s); {len(running)} total live.")
    _beat(sum(1 for th in running.values() if th.is_alive()))


def start_iq_option_engine(on_candle: CandleCallback = None,
                           rescan_interval: int = RESCAN_INTERVAL) -> threading.Thread:
    """
    Launch the IQ Option engine on a background daemon thread. Safe to call
    unconditionally: if the library isn't installed or no users have
    connected, it just idles and logs. Returns the engine thread.
    """
    def _loop_forever():
        running: dict = {}
        stop_event = threading.Event()
        logger.info("[iq_option] engine thread launched.")
        while True:
            try:
                _rescan_and_launch(on_candle, running, stop_event)
            except Exception as exc:
                logger.error(f"[iq_option] engine cycle error: {exc}", exc_info=True)
            time.sleep(rescan_interval)

    t = threading.Thread(target=_loop_forever, name="IQOptionEngine", daemon=True)
    t.start()
    return t

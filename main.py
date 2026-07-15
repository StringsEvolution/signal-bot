"""
Main Orchestrator
Runs the signal bot in a loop:
  - Refresh OHLC data in a BACKGROUND THREAD (non-blocking)
  - Scan IMMEDIATELY when fresh data is ready — no waiting
  - Send qualifying signals to Telegram instantly
  - Generate daily/weekly reports on schedule
  - Poll Telegram for bot commands
  - Reset Twelve Data API keys at midnight UTC

Usage:
  python main.py                  # run live bot
  python main.py --backtest       # run full backtest
  python main.py --scan-once      # single scan and exit
"""

import os
import sys
import time
import threading
import logging
import argparse
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("signal_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def run_live_bot():
    """
    Main loop — event-driven, not polling.

    Architecture:
    - One persistent Deriv WebSocket per asset, subscribed live to every
      timeframe at once (data_engine.run_streaming_engine).
    - The instant a candle closes on ANY timeframe, the asyncio stream
      fires `_on_candle_close` immediately — no waiting for a shared
      refresh timer.
    - Candle storage + signal scan + Telegram send for that one
      asset/timeframe run in a worker thread so they never block the
      WebSocket event loop (which is juggling 5 live connections).
    - Reports / midnight reset run on their own lightweight ticker thread,
      independent of the streaming loop.

    Result: total latency from real candle close to Telegram delivery is
    bounded by network + processing time only — typically 2-5 seconds,
    for every timeframe, every time.
    """
    import asyncio
    import concurrent.futures
    import pandas as pd

    from data_engine   import (
        init_db, refresh_all, get_key_manager, store_ohlc,
        run_streaming_engine, ASSETS, TIMEFRAMES, TF_MINUTES,
        DBUnavailable,
    )
    from signal_engine import generate_signal
    from telegram_bot  import send_signal, send_otc_signal, send_admin_alert, BotCommandHandler
    from performance_tracker import generate_daily_report, generate_weekly_report, log_signal
    from settlement import worker as settlement_worker
    from prealert import worker as prealert_worker

    logger.info("=" * 60)
    logger.info("  Signal Bot Pro — Starting (real-time streaming mode)")
    logger.info("=" * 60)

    try:
        init_db()
    except DBUnavailable as exc:
        # The DB is down for a hard reason (most commonly: Neon/Postgres
        # transfer-quota exceeded). Crashing here just makes the platform
        # restart us in ~2s, and every restart re-attempts the connection —
        # which BURNS MORE of the very quota that's blocking us. So instead we
        # hold the process alive and idle. No reconnect churn, no restart
        # storm. Fix the quota (upgrade plan / reduce transfer / wait for
        # monthly reset), then redeploy or restart once.
        logger.critical("=" * 60)
        logger.critical("STARTUP HALTED — database unreachable.")
        logger.critical(str(exc))
        logger.critical(
            "If this is a Neon 'data transfer quota exceeded' error: the bot "
            "will NOT auto-recover by restarting. Resolve the quota, then "
            "restart this service once."
        )
        logger.critical("Holding idle to avoid a restart/reconnect storm...")
        logger.critical("=" * 60)
        try:
            send_admin_alert(
                "⛔ Signal Bot could not start: database unreachable "
                "(likely Postgres transfer-quota exceeded). Holding idle. "
                "Fix the DB quota, then restart the service."
            )
        except Exception:
            pass
        # Idle forever (until operator restarts). Sleep in a loop so signals
        # like SIGTERM still terminate us cleanly on redeploy.
        while True:
            time.sleep(3600)

    from user_manager import init_user_tables, init_platform_tables
    init_user_tables()
    init_platform_tables()
    send_admin_alert("🚀 Signal Bot Pro is now online and streaming markets live.")

    cmd_handler = BotCommandHandler()
    cmd_handler.start()

    # Start the settlement worker (BUG B): scores each fired signal win/loss
    # once its expiry elapses, so daily/weekly reports actually populate.
    settlement_worker.start()

    # Pre-alert worker: heads-up a few seconds before a signal confirms.
    # Disabled by default (it was causing channel noise). Re-enable any time
    # by setting PREALERT_ENABLED=1 in the environment — no code change needed.
    if os.getenv("PREALERT_ENABLED", "0") == "1":
        prealert_worker.start()
        logger.info("Pre-alert worker started (PREALERT_ENABLED=1).")
    else:
        logger.info("Pre-alert worker disabled (set PREALERT_ENABLED=1 to enable).")

    # ----------------------------------------------------------------
    # Startup seed — runs ONCE with the new parallel fetch (~8s).
    # Hard 90s timeout: if DB/network stalls we still start streaming.
    # ----------------------------------------------------------------
    logger.info("Seeding initial OHLC history...")
    seed_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    seed_future = seed_pool.submit(refresh_all)
    try:
        seed_future.result(timeout=90)
        logger.info("Initial seed complete.")
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Initial seed did not finish within 90s — starting live streams "
            "anyway. Backfill loop will pick up any missing history."
        )
    except Exception as exc:
        logger.error(f"Initial seed refresh failed: {exc} — starting live streams anyway.")
    seed_pool.shutdown(wait=False)

    worker_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=25, thread_name_prefix="signal-worker"
    )

    # ----------------------------------------------------------------
    # Pocket Option (OTC) — per-user streams, fully additive.
    #
    # Runs on its own background thread + event loop so it never
    # competes with the Deriv streaming loop above. store_ohlc() writes
    # Pocket Option candles under an "_otc" suffixed asset code (e.g.
    # "EURUSD_otc") into the SAME ohlc_data table Deriv uses, so
    # generate_signal() below works identically for both sources.
    #
    # If the `pocket-option` package isn't installed, or no user has
    # connected an account yet (via /connectpo), this quietly no-ops —
    # it never affects the Deriv signal pipeline.
    # ----------------------------------------------------------------
    # Dedup guard: several users can stream the SAME OTC asset, so this
    # handler may fire multiple times for one candle. We only want to
    # generate + deliver the signal once per (asset, timeframe, candle).
    _otc_seen: dict = {}
    _otc_seen_lock = threading.Lock()

    def _handle_po_candle_close(asset: str, tf: str, candle: dict):
        try:
            # --- Deduplicate across concurrent user streams ---
            key    = f"{asset}|{tf}|{candle['open_time']}"
            now_ts = time.time()
            with _otc_seen_lock:
                # Evict stale keys (older than 2 minutes) so this stays small.
                for k in [k for k, v in _otc_seen.items() if now_ts - v > 120]:
                    _otc_seen.pop(k, None)
                if key in _otc_seen:
                    return  # already handled this exact OTC candle
                _otc_seen[key] = now_ts

            close_epoch = candle["open_time"] + TF_MINUTES[tf] * 60
            close_dt    = datetime.fromtimestamp(close_epoch, tz=timezone.utc)
            now         = datetime.now(timezone.utc)

            sig = generate_signal(asset, tf, dt=now)
            if sig:
                logger.info(f"  → OTC SIGNAL: {asset}/{tf} {sig.direction} conf={sig.confidence:.0f}%")
                # OTC signals go PRIVATELY into the bot (per-user DMs),
                # NOT to the VIP/Free channels. Non-OTC Deriv signals
                # continue to use send_signal() → VIP channel unchanged.
                send_otc_signal(sig)
                try:
                    sig_id = log_signal(sig)
                    if sig_id is not None:
                        settlement_worker.log_pending(sig, sig_id)
                except Exception as log_exc:
                    logger.error(f"OTC signal logging/settlement queue failed: {log_exc}")
        except Exception as exc:
            logger.error(f"Pocket Option candle-close handler error {asset}/{tf}: {exc}", exc_info=True)

    def _on_po_candle(asset: str, tf: str, candle: dict):
        worker_pool.submit(_handle_po_candle_close, asset, tf, candle)

    po_engine_thread = None
    try:
        from pocket_option_engine import start_pocket_option_engine
        po_engine_thread = start_pocket_option_engine(on_candle=_on_po_candle)
    except Exception as exc:
        logger.warning(f"Pocket Option engine could not be started: {exc}")

    # IQ Option OTC engine. Reuses _on_po_candle: IQ candles are stored under
    # the SAME internal `_otc` codes and flow through the identical
    # _handle_po_candle_close -> send_otc_signal path, so OTC signals from IQ
    # are delivered privately per-user exactly like Pocket Option's.
    iq_engine_thread = None
    try:
        from iq_option_engine import start_iq_option_engine
        iq_engine_thread = start_iq_option_engine(on_candle=_on_po_candle)
    except Exception as exc:
        logger.warning(f"IQ Option engine could not be started: {exc}")

    # ----------------------------------------------------------------
    # Fires the instant a candle closes for one asset/timeframe.
    # Runs in a worker thread — never blocks the live WS connections.
    # ----------------------------------------------------------------
    def _handle_candle_close(asset: str, tf: str, candle: dict):
        try:
            handler_start = datetime.now(timezone.utc)

            close_epoch = candle["open_time"] + TF_MINUTES[tf] * 60
            close_dt    = datetime.fromtimestamp(close_epoch, tz=timezone.utc)
            detect_lag  = (handler_start - close_dt).total_seconds()

            ts = pd.to_datetime(candle["open_time"], unit="s")
            df = pd.DataFrame([{
                "timestamp": ts,
                "open":  candle["open"],
                "high":  candle["high"],
                "low":   candle["low"],
                "close": candle["close"],
                "volume": 0.0,
            }])
            store_ohlc(asset, tf, df)

            now = datetime.now(timezone.utc)
            sig = generate_signal(asset, tf, dt=now)

            if sig:
                pre_send_lag = (datetime.now(timezone.utc) - close_dt).total_seconds()
                logger.info(
                    f"  → SIGNAL: {asset}/{tf} {sig.direction} "
                    f"conf={sig.confidence:.0f}% | "
                    f"close={close_dt.strftime('%H:%M:%S')} UTC | "
                    f"detect_lag={detect_lag:.2f}s pre_send_lag={pre_send_lag:.2f}s"
                )
                send_start = datetime.now(timezone.utc)
                send_signal(sig)
                send_done   = datetime.now(timezone.utc)

                # Persist the signal and queue it for result settlement (BUG B).
                # log_signal writes the row (result=NULL) and returns its id;
                # the settlement worker fills in win/loss after expiry.
                try:
                    sig_id = log_signal(sig)
                    if sig_id is not None:
                        settlement_worker.log_pending(sig, sig_id)
                except Exception as log_exc:
                    logger.error(f"Signal logging/settlement queue failed: {log_exc}")

                total_lag   = (send_done - close_dt).total_seconds()
                telegram_ms = (send_done - send_start).total_seconds()
                logger.info(
                    f"  ✓ DELIVERED: {asset}/{tf} | "
                    f"telegram_call={telegram_ms:.2f}s | "
                    f"TOTAL candle-close→Telegram lag={total_lag:.2f}s"
                )
            else:
                if detect_lag > 5:
                    logger.warning(
                        f"  ⚠ {asset}/{tf}: detect_lag={detect_lag:.2f}s "
                        f"(candle closed {close_dt.strftime('%H:%M:%S')} UTC) — "
                        f"higher than expected, check stream health"
                    )
        except Exception as exc:
            logger.error(f"Candle-close handler error {asset}/{tf}: {exc}", exc_info=True)

    async def _on_close(asset: str, tf: str, candle: dict):
        loop = asyncio.get_running_loop()
        loop.run_in_executor(worker_pool, _handle_candle_close, asset, tf, candle)

    # ----------------------------------------------------------------
    # Scheduler — midnight reset + daily/weekly reports (every 60s tick)
    # ----------------------------------------------------------------
    _stop_event = threading.Event()
    state = {
        "last_daily_report":   datetime.now(timezone.utc),
        "last_weekly_report":  datetime.now(timezone.utc),
        "last_midnight_reset": datetime.now(timezone.utc),
    }

    def _scheduler_loop():
        while not _stop_event.is_set():
            _stop_event.wait(timeout=60)
            if _stop_event.is_set():
                break
            try:
                now = datetime.now(timezone.utc)

                if now.hour == 0 and now.minute == 0 and \
                        (now - state["last_midnight_reset"]).total_seconds() > 3600:
                    get_key_manager().reset_daily()
                    state["last_midnight_reset"] = now
                    logger.info("Midnight UTC — Twelve Data API keys reset.")

                if now.hour == 22 and (now - state["last_daily_report"]).total_seconds() > 3600:
                    try:
                        report = generate_daily_report()
                        from telegram_bot import send_performance_report
                        send_performance_report(report, "Daily")
                        state["last_daily_report"] = now
                        logger.info("Daily report sent.")
                    except Exception as exc:
                        logger.error(f"Daily report failed: {exc}")

                if now.weekday() == 6 and now.hour == 22 and \
                        (now - state["last_weekly_report"]).total_seconds() > 3600:
                    try:
                        report = generate_weekly_report()
                        from telegram_bot import send_performance_report
                        send_performance_report(report, "Weekly")
                        state["last_weekly_report"] = now
                        logger.info("Weekly report sent.")
                    except Exception as exc:
                        logger.error(f"Weekly report failed: {exc}")
            except Exception as exc:
                logger.error(f"Scheduler tick error: {exc}", exc_info=True)

    threading.Thread(target=_scheduler_loop, name="Scheduler", daemon=True).start()

    # ----------------------------------------------------------------
    # Backfill safety net — runs every 30 MINUTES (not every 5 minutes).
    #
    # WHY CHANGED:
    # - Old interval: 300s (5 min). Old fetch time: ~320s sequential.
    #   Result: backfill was running back-to-back non-stop, consuming
    #   all DB connections and hammering Deriv WS constantly.
    # - New interval: 1800s (30 min). New fetch time: ~8s parallel.
    #   Result: backfill is a rare, fast gap-repair — not a hot loop.
    #
    # The live streams (run_streaming_engine) keep data fresh in real
    # time. This backfill only exists to recover from a dropped stream
    # or a Railway container restart. 30 minutes is more than enough.
    # ----------------------------------------------------------------
    _last_backfill = {"time": datetime.now(timezone.utc)}

    def _backfill_loop():
        while not _stop_event.is_set():
            # Sleep 30 minutes between backfills
            _stop_event.wait(timeout=1800)
            if _stop_event.is_set():
                break
            try:
                logger.info("Backfill: running scheduled gap-repair refresh...")
                t0 = time.time()
                refresh_all()
                elapsed = time.time() - t0
                _last_backfill["time"] = datetime.now(timezone.utc)
                logger.info(f"Backfill: complete in {elapsed:.1f}s")
            except Exception as exc:
                logger.error(f"Backfill refresh failed: {exc}")

    threading.Thread(target=_backfill_loop, name="Backfill", daemon=True).start()

    # ----------------------------------------------------------------
    # Pocket Option watchdog — observational + self-heal.
    #
    # Checks the OTC engine's heartbeat every 2 minutes. If the engine
    # thread was started and then either DIES (auto-restart it) or STALLS
    # (alive but not cycling — alert admin), it's surfaced instead of
    # failing silently. This is entirely OTC-scoped: the Deriv streaming
    # loop below is never affected, and if the SDK isn't installed the
    # watchdog stays quiet (engine never reports "started").
    # ----------------------------------------------------------------
    def _po_watchdog_loop():
        nonlocal po_engine_thread
        try:
            from pocket_option_engine import get_engine_health, start_pocket_option_engine
        except Exception as exc:
            logger.warning(f"[watchdog] Pocket Option health unavailable: {exc}")
            return

        STALE_AFTER = 900   # seconds — comfortably beyond 2 rescan cycles (300s)
        stale_alerted = False

        while not _stop_event.is_set():
            _stop_event.wait(timeout=120)
            if _stop_event.is_set():
                break
            try:
                if po_engine_thread is None:
                    continue

                health = get_engine_health()
                if not health.get("started"):
                    # SDK absent or engine hasn't begun cycling yet — not a fault.
                    continue

                # Case 1: thread died outright — try to bring it back.
                if not po_engine_thread.is_alive():
                    logger.error("[watchdog] Pocket Option engine thread died — attempting restart.")
                    try:
                        po_engine_thread = start_pocket_option_engine(on_candle=_on_po_candle)
                        send_admin_alert(
                            "♻️ Pocket Option engine thread had stopped and was "
                            "auto-restarted. Deriv signals were unaffected."
                        )
                    except Exception as exc:
                        logger.error(f"[watchdog] restart failed: {exc}")
                        send_admin_alert(f"⚠️ Pocket Option engine died and restart failed: {exc}")
                    continue

                # Case 2: thread alive but heartbeat is stale (stuck cycle).
                age = health.get("age_seconds")
                if age is not None and age > STALE_AFTER:
                    if not stale_alerted:
                        logger.error(f"[watchdog] Pocket Option heartbeat stale ({age}s).")
                        send_admin_alert(
                            f"⚠️ Pocket Option engine hasn't cycled in {age:.0f}s — "
                            f"OTC signals may be paused. Deriv signals unaffected."
                        )
                        stale_alerted = True
                elif stale_alerted:
                    logger.info("[watchdog] Pocket Option engine recovered.")
                    send_admin_alert("✅ Pocket Option engine recovered — OTC streaming healthy again.")
                    stale_alerted = False
            except Exception as exc:
                logger.error(f"[watchdog] loop error: {exc}", exc_info=True)

    threading.Thread(target=_po_watchdog_loop, name="PocketOptionWatchdog", daemon=True).start()

    logger.info(f"Opening live streams for {len(ASSETS)} assets × {len(TIMEFRAMES)} timeframes...")

    try:
        asyncio.run(run_streaming_engine(_on_close))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        _stop_event.set()
        cmd_handler.stop()
        send_admin_alert("⏹️ Signal Bot Pro was stopped manually.")
    except Exception as exc:
        logger.error(f"Streaming engine crashed: {exc}", exc_info=True)
        _stop_event.set()
        cmd_handler.stop()
        send_admin_alert(f"⚠️ Signal Bot Pro crashed: {exc}")


def run_backtest():
    from backtester import Backtester
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Starting full backtest...")
    bt     = Backtester()
    result = bt.run_all(n_candles=500)

    print(result.summary())

    if result.equity_curve:
        plt.figure(figsize=(12, 5))
        plt.plot(result.equity_curve, color="#00D4FF", linewidth=1.5)
        plt.axhline(100, color="#64748B", linestyle="--", linewidth=0.8)
        plt.fill_between(range(len(result.equity_curve)),
                         result.equity_curve, 100,
                         where=[v >= 100 for v in result.equity_curve],
                         alpha=0.15, color="#00E5A0")
        plt.fill_between(range(len(result.equity_curve)),
                         result.equity_curve, 100,
                         where=[v < 100 for v in result.equity_curve],
                         alpha=0.15, color="#FF4D6D")
        plt.title("Backtest Equity Curve")
        plt.xlabel("Trade #")
        plt.ylabel("Equity (%)")
        plt.tight_layout()
        plt.savefig("backtest_equity.png", dpi=150)
        logger.info("Equity curve saved to backtest_equity.png")

    return result


def run_single_scan():
    from data_engine   import init_db, refresh_all
    from signal_engine import scan_all
    from telegram_bot  import send_signal

    init_db()
    refresh_all()

    signals = scan_all()

    if not signals:
        print("\n❌ No qualifying signals found in this scan.\n")
        print("Possible reasons:")
        print("  • Market is ranging or in a dead session")
        print("  • No confirmed S/R + pattern + indicator confluence")
        print("  • AI confidence below threshold")
        return

    print(f"\n✅ {len(signals)} signal(s) found:\n")
    print("=" * 60)

    for sig in signals:
        print(sig.format_message(vip=True))
        print("=" * 60)
        send_signal(sig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Bot Pro")
    parser.add_argument("--backtest",  action="store_true", help="Run full backtest")
    parser.add_argument("--scan-once", action="store_true", help="Single scan and exit")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.scan_once:
        run_single_scan()
    else:
        run_live_bot()

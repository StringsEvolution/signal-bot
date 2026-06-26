"""
Main Orchestrator
Runs the signal bot in a loop:
  - Refresh OHLC data every minute
  - Scan IMMEDIATELY after refresh completes (fresh data)
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
    Main loop flow:
      1. Refresh OHLC data (parallel, ~45s)
      2. Scan for signals IMMEDIATELY after refresh — freshest data
      3. Send signals to Telegram instantly
      4. Sleep only the remaining time to fill 60s cycle
      5. Repeat
    """
    from data_engine   import init_db, refresh_all, get_key_manager
    from signal_engine import scan_all
    from telegram_bot  import send_signal, send_admin_alert, BotCommandHandler
    from performance_tracker import generate_daily_report, generate_weekly_report

    logger.info("=" * 60)
    logger.info("  Signal Bot Pro — Starting")
    logger.info("=" * 60)

    init_db()
    send_admin_alert("🚀 Signal Bot Pro is now online and scanning markets.")

    cmd_handler = BotCommandHandler()
    cmd_handler.start()

    last_daily_report  = datetime.now(timezone.utc)
    last_weekly_report = datetime.now(timezone.utc)
    last_midnight_reset = datetime.now(timezone.utc)

    CYCLE_INTERVAL = int(os.getenv("SCAN_INTERVAL_SEC", "60"))

    while True:
        cycle_start = datetime.now(timezone.utc)

        try:
            now = cycle_start

            # 1. Midnight reset
            if now.hour == 0 and now.minute == 0 and (now - last_midnight_reset).seconds > 3600:
                get_key_manager().reset_daily()
                last_midnight_reset = now
                logger.info("Midnight UTC — Twelve Data API keys reset.")

            # 2. Refresh OHLC data — fetch all pairs in parallel
            logger.info("Refreshing OHLC data...")
            try:
                refresh_all()
            except Exception as exc:
                logger.error(f"Data refresh failed: {exc}")

            # 3. Scan IMMEDIATELY after refresh — data is as fresh as possible
            scan_time = datetime.now(timezone.utc)
            logger.info("Scanning for signals...")
            signals = scan_all(dt=scan_time)

            # 4. Send signals INSTANTLY — no delay between scan and send
            if signals:
                for sig in signals:
                    logger.info(f"  → SIGNAL: {sig.asset}/{sig.timeframe} {sig.direction} conf={sig.confidence:.0f}%")
                    send_signal(sig)
            else:
                logger.info("  No qualifying signals this scan.")

            # 5. Daily report at 22:00 UTC
            if now.hour == 22 and (now - last_daily_report).total_seconds() > 3600:
                try:
                    report = generate_daily_report()
                    from telegram_bot import send_performance_report
                    send_performance_report(report, "Daily")
                    last_daily_report = now
                    logger.info("Daily report sent.")
                except Exception as exc:
                    logger.error(f"Daily report failed: {exc}")

            # 6. Weekly report Sunday 22:00 UTC
            if now.weekday() == 6 and now.hour == 22 and (now - last_weekly_report).total_seconds() > 3600:
                try:
                    report = generate_weekly_report()
                    from telegram_bot import send_performance_report
                    send_performance_report(report, "Weekly")
                    last_weekly_report = now
                    logger.info("Weekly report sent.")
                except Exception as exc:
                    logger.error(f"Weekly report failed: {exc}")

            # 7. Sleep only remaining time to fill the cycle
            # This ensures the next refresh starts exactly CYCLE_INTERVAL seconds
            # after the previous one started — not after it ended
            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            sleep_time = max(0, CYCLE_INTERVAL - elapsed)
            logger.info(f"Cycle took {elapsed:.1f}s. Sleeping {sleep_time:.1f}s until next scan...")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            cmd_handler.stop()
            send_admin_alert("⏹️ Signal Bot Pro was stopped manually.")
            break
        except Exception as exc:
            logger.error(f"Main loop error: {exc}", exc_info=True)
            time.sleep(30)


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

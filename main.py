"""
Main Orchestrator
Runs the signal bot in a loop:
  - Refresh OHLC data every minute
  - Scan for signals on every refresh
  - Send qualifying signals to Telegram
  - Generate daily/weekly reports on schedule
  - Poll Telegram for bot commands

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
from datetime import datetime, timedelta
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
    """Main loop: data refresh → scan → Telegram → sleep."""
    from data_engine   import init_db, refresh_all
    from signal_engine import scan_all
    from telegram_bot  import send_signal, send_admin_alert, BotCommandHandler
    from performance_tracker import generate_daily_report, generate_weekly_report

    logger.info("=" * 60)
    logger.info("  Signal Bot Pro — Starting")
    logger.info("=" * 60)

    init_db()
    send_admin_alert("🚀 Signal Bot Pro is now online and scanning markets.")

    cmd_handler       = BotCommandHandler()
    cmd_handler.start()   # launches background thread — commands respond instantly
    last_data_refresh = datetime.min
    last_daily_report = datetime.min
    last_weekly_report = datetime.min

    SCAN_INTERVAL    = int(os.getenv("SCAN_INTERVAL_SEC",    "60"))
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL_SEC", "60"))

    while True:
        try:
            now = datetime.utcnow()

            # 1. Refresh OHLC data
            if (now - last_data_refresh).seconds >= REFRESH_INTERVAL:
                logger.info("Refreshing OHLC data...")
                try:
                    refresh_all()
                    last_data_refresh = now
                except Exception as exc:
                    logger.error(f"Data refresh failed: {exc}")

            # 2. Scan for signals
            logger.info("Scanning for signals...")
            signals = scan_all(dt=now)

            for sig in signals:
                logger.info(f"  → SIGNAL: {sig.asset}/{sig.timeframe} {sig.direction} conf={sig.confidence:.0f}%")
                send_signal(sig)

            if not signals:
                logger.info("  No qualifying signals this scan.")

            # 3. Daily report at 22:00 UTC
            if now.hour == 22 and (now - last_daily_report).seconds > 3600:
                report = generate_daily_report()
                from telegram_bot import send_performance_report
                send_performance_report(report, "Daily")
                last_daily_report = now
                logger.info("Daily report sent.")

            # 4. Weekly report Sunday 22:00 UTC
            if now.weekday() == 6 and now.hour == 22 and (now - last_weekly_report).seconds > 3600:
                report = generate_weekly_report()
                from telegram_bot import send_performance_report
                send_performance_report(report, "Weekly")
                last_weekly_report = now
                logger.info("Weekly report sent.")

            logger.info(f"Sleeping {SCAN_INTERVAL}s until next scan...")
            time.sleep(SCAN_INTERVAL)

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

    # Plot equity curve
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
        print("  • AI confidence below 80%")
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

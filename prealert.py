"""
Pre-alert worker
----------------
Gives users a heads-up a few seconds BEFORE a signal confirms, so they have
time to open Pocket Option and get ready.

How it works (and its honest limitation):
  A signal is computed from the candle that just CLOSED. So there is no way to
  know for certain that a signal is coming until the candle closes. What we can
  do is peek at the still-FORMING candle a few seconds before it closes, run the
  exact same gate stack against it (signal_engine.generate_signal(preview=True)),
  and if it currently passes, send a provisional "get ready" heads-up. The real
  signal — or nothing, if the last seconds change the setup — follows at close.

  So a pre-alert is a "set up now", not a promise. Some pre-alerts won't convert
  into a confirmed signal. The heads-up text says so explicitly.

Design:
  - One background thread. Every POLL_SECS it scans each asset/timeframe.
  - For each, it reads the forming candle from data_engine.get_forming_candle().
  - If that candle is within LEAD_SECS of closing AND we haven't already
    pre-alerted THIS candle, it runs preview evaluation. If a signal is returned,
    it broadcasts one heads-up and records the candle so it won't repeat.
  - No DB writes, no effect on the real signal path.
"""

import time
import logging
import threading
from datetime import datetime, timezone

from data_engine import get_forming_candle, ASSETS, TIMEFRAMES, TF_MINUTES
from signal_engine import generate_signal
from telegram_bot import send_prealert

logger = logging.getLogger(__name__)

# How often the worker scans for candles about to close.
POLL_SECS = 3

# How many seconds before a candle closes we send the heads-up.
# Big enough to open the platform; small enough that the forming candle is
# already close to its final shape.
LEAD_SECS = 10

# Don't fire a heads-up if the candle is already within this many seconds of
# close — too late to be useful, and the confirmed signal is imminent anyway.
MIN_LEAD_SECS = 3


class PreAlertWorker:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        # (asset, tf, candle_open_time) already pre-alerted → avoid duplicates
        self._alerted: set = set()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="prealert-worker", daemon=True
        )
        self._thread.start()
        logger.info("Pre-alert worker started.")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._scan()
            except Exception as exc:
                logger.error(f"Pre-alert scan error: {exc}", exc_info=True)
            self._stop.wait(POLL_SECS)

    def _scan(self):
        now_epoch = datetime.now(timezone.utc).timestamp()

        for asset in ASSETS:
            for tf in TIMEFRAMES:
                snap = get_forming_candle(asset, tf)
                if not snap:
                    continue

                open_time  = int(snap["open_time"])
                close_time = open_time + TF_MINUTES[tf] * 60
                seconds_left = close_time - now_epoch

                # Only in the lead window
                if seconds_left > LEAD_SECS or seconds_left < MIN_LEAD_SECS:
                    continue

                key = (asset, tf, open_time)
                if key in self._alerted:
                    continue

                # Run the real gate stack against the forming candle, no side effects
                try:
                    sig = generate_signal(
                        asset, tf, preview=True, forming_candle=snap
                    )
                except Exception as exc:
                    logger.warning(f"Pre-alert preview failed {asset}/{tf}: {exc}")
                    self._alerted.add(key)  # don't retry this candle
                    continue

                # Mark handled regardless, so we send at most one heads-up per candle
                self._alerted.add(key)

                if sig is not None:
                    logger.info(
                        f"⏰ PRE-ALERT: {asset}/{tf} {sig.direction} "
                        f"conf={sig.confidence:.0f}% (~{int(seconds_left)}s to close)"
                    )
                    try:
                        send_prealert(
                            asset, tf, sig.direction,
                            sig.confidence, int(seconds_left)
                        )
                    except Exception as exc:
                        logger.error(f"Pre-alert send failed {asset}/{tf}: {exc}")

        self._prune()

    def _prune(self):
        # Keep the dedup set from growing forever: drop entries for candles that
        # closed more than 2 minutes ago.
        if len(self._alerted) < 500:
            return
        now_epoch = datetime.now(timezone.utc).timestamp()
        self._alerted = {
            (a, tf, ot) for (a, tf, ot) in self._alerted
            if (ot + TF_MINUTES.get(tf, 1) * 60) > (now_epoch - 120)
        }


worker = PreAlertWorker()

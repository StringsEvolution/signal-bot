"""
Telegram Bot — Sends signals to Free and VIP channels.
Free channel: direction + confidence only.
VIP channel: full signal with all reasons.

Command handler runs in a dedicated background thread so /start, /status, /stats
respond instantly without waiting for the scan loop.
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
FREE_CHANNEL   = os.getenv("TELEGRAM_FREE_CHANNEL",  "")
VIP_CHANNEL    = os.getenv("TELEGRAM_VIP_CHANNEL",   "")
ADMIN_CHAT_ID  = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# ---------------------------------------------------------------------------
# Low-level sender
# ---------------------------------------------------------------------------

def _send_message(chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    if not BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured — message not sent.")
        return False

    url  = TELEGRAM_API.format(token=BOT_TOKEN, method="sendMessage")
    data = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.ok:
            logger.info(f"Telegram message sent to {chat_id}")
            return True
        else:
            logger.error(f"Telegram error: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Signal formatters
# ---------------------------------------------------------------------------

def _format_free_message(signal) -> str:
    icon = "🟢" if signal.direction == "CALL" else "🔴"
    return (
        f"🔔 <b>New Signal</b>\n\n"
        f"{icon} <b>{signal.direction}</b> — {signal.asset}\n"
        f"⏳ Expiry: {signal.expiry_min} min\n"
        f"🤖 Confidence: <b>{signal.confidence:.0f}%</b>\n"
        f"🕒 Time: {signal.timestamp.strftime('%H:%M UTC')}\n\n"
        f"📊 Full analysis available in VIP 👇\n"
        f"🔒 Join: {os.getenv('VIP_INVITE_LINK', 't.me/your_vip_link')}"
    )


def _format_vip_message(signal) -> str:
    icon = "🟢" if signal.direction == "CALL" else "🔴"
    reasons_html = "\n".join(f"  ✅ {r}" for r in signal.reasons)
    warn_html    = ""
    if signal.warnings:
        warn_html = "\n⚠️ <b>Warnings:</b>\n" + "\n".join(f"  ⚠️ {w}" for w in signal.warnings)

    return (
        f"⭐ <b>VIP Signal</b> — {icon} {signal.direction}\n\n"
        f"📊 <b>Pair:</b>      {signal.asset}\n"
        f"📈 <b>Timeframe:</b> {signal.timeframe}\n"
        f"⏳ <b>Expiry:</b>    {signal.expiry_min} minutes\n"
        f"💰 <b>Entry:</b>     <code>{signal.entry_price:.5f}</code>\n"
        f"🤖 <b>Confidence:</b> <b>{signal.confidence:.0f}%</b>\n"
        f"🌍 <b>Session:</b>   {signal.session}\n"
        f"🕒 <b>UTC Time:</b>  {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"📋 <b>Analysis:</b>\n{reasons_html}{warn_html}\n\n"
        f"⚠️ <i>Risk disclaimer: Binary options carry significant financial risk. "
        f"Never trade with money you cannot afford to lose. Past performance does not guarantee future results.</i>"
    )


# ---------------------------------------------------------------------------
# Public send functions
# ---------------------------------------------------------------------------

def send_signal(signal) -> dict:
    results = {}
    if FREE_CHANNEL:
        results["free"] = _send_message(FREE_CHANNEL, _format_free_message(signal))
    if VIP_CHANNEL:
        results["vip"]  = _send_message(VIP_CHANNEL,  _format_vip_message(signal))
    return results


def send_admin_alert(text: str):
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, f"🤖 <b>Bot Alert</b>\n\n{text}")


def send_performance_report(report_text: str, report_type: str = "Daily"):
    msg = (
        f"📊 <b>{report_type} Performance Report</b>\n"
        f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{report_text}"
    )
    if VIP_CHANNEL:
        _send_message(VIP_CHANNEL, msg)
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, msg)


# ---------------------------------------------------------------------------
# Command handler — runs in its own thread, long-polls continuously
# ---------------------------------------------------------------------------

class BotCommandHandler:
    """
    Runs in a dedicated background thread using long-polling (timeout=25s).
    Commands are processed immediately as they arrive — no scan-loop delay.
    """

    def __init__(self):
        self.offset  = 0
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    # ---- Public API --------------------------------------------------------

    def start(self):
        """Start the background polling thread. Call once from main."""
        if not BOT_TOKEN:
            logger.warning("No TELEGRAM_BOT_TOKEN — command handler disabled.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="TelegramPoller",
            daemon=True,       # dies automatically when main process exits
        )
        self._thread.start()
        logger.info("Telegram command handler started (background thread).")

    def stop(self):
        """Signal the polling thread to stop."""
        self._stop.set()

    def poll_once(self):
        """
        Legacy compatibility shim — no-op if background thread is running,
        otherwise does a single synchronous poll (used in tests).
        """
        if self._thread and self._thread.is_alive():
            return   # background thread handles it
        self._do_poll()

    # ---- Internal ----------------------------------------------------------

    def _poll_loop(self):
        """Continuous long-poll loop running in background thread."""
        logger.info("Telegram poll loop running.")
        while not self._stop.is_set():
            try:
                self._do_poll()
            except Exception as exc:
                logger.error(f"Poll loop error: {exc}")
                time.sleep(5)   # brief back-off on error, then retry

    def _do_poll(self):
        """
        Single long-poll call.
        timeout=25 means Telegram holds the connection open for up to 25s
        waiting for updates before returning an empty list.
        This gives near-instant command responses without hammering the API.
        """
        if not BOT_TOKEN:
            return

        url  = TELEGRAM_API.format(token=BOT_TOKEN, method="getUpdates")
        data = {
            "offset":  self.offset,
            "timeout": 25,      # long-poll: waits up to 25s for new messages
            "limit":   10,
        }
        try:
            resp = requests.get(url, params=data, timeout=30)  # slightly > telegram timeout
            if not resp.ok:
                logger.warning(f"getUpdates returned {resp.status_code}")
                time.sleep(3)
                return

            updates = resp.json().get("result", [])
            for update in updates:
                self.offset = update["update_id"] + 1
                try:
                    self._handle_update(update)
                except Exception as exc:
                    logger.error(f"Handle update error: {exc}")

        except requests.exceptions.Timeout:
            pass   # normal — long-poll expired with no messages
        except Exception as exc:
            logger.error(f"Poll error: {exc}")
            time.sleep(3)

    def _handle_update(self, update: dict):
        msg = update.get("message", {})
        if not msg:
            return

        chat_id  = str(msg["chat"]["id"])
        text     = msg.get("text", "").strip()
        username = msg.get("from", {}).get("username", "unknown")

        logger.info(f"Command from @{username}: {text}")

        if text.startswith("/start"):
            _send_message(chat_id,
                "👋 Welcome to <b>Signal Bot Pro</b>!\n\n"
                "I generate high-confidence binary options signals using "
                "AI + market structure analysis.\n\n"
                "<b>Commands:</b>\n"
                "/status — Bot status &amp; uptime\n"
                "/stats  — Today's performance\n"
                "/pairs  — Scanned assets &amp; timeframes\n"
                "/help   — Show this menu\n\n"
                "⚠️ <i>Trading involves significant risk. Never invest more than you can afford to lose.</i>"
            )

        elif text.startswith("/status"):
            uptime = _get_uptime()
            _send_message(chat_id,
                f"✅ <b>Bot Status: ONLINE</b>\n"
                f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"⏱️ Uptime: {uptime}\n\n"
                f"📡 <b>Scanning:</b>\n"
                f"  Pairs: EURUSD · GBPUSD · XAUUSD . USDJPY . BTC USD\n"
                f"  Timeframes: M1 · M5 · M15\n"
                f"  Interval: every 60s\n\n"
                f"🤖 AI mode: {'ML model' if _ml_model_loaded() else 'Heuristic (pre-training)'}\n"
                f"🔒 Confidence threshold: 80%"
            )

        elif text.startswith("/stats"):
            try:
                from performance_tracker import generate_daily_report
                report = generate_daily_report()
                _send_message(chat_id, report)
            except Exception as exc:
                _send_message(chat_id,
                    "📊 <b>Today's Stats</b>\n\n"
                    "No completed signals yet today, or database not connected.\n"
                    "Stats update after signals expire and results are logged."
                )

        elif text.startswith("/pairs"):
            _send_message(chat_id,
                "📊 <b>Monitored Pairs</b>\n\n"
                "🔵 <b>EURUSD</b> — Euro / US Dollar\n"
                "🔵 <b>GBPUSD</b> — British Pound / US Dollar\n"
                "🟡 <b>XAUUSD</b> — Gold / US Dollar\n\n"
                "<b>Timeframes &amp; Expiry:</b>\n"
                "  M1  → 3 min expiry\n"
                "  M5  → 5 min expiry\n"
                "  M15 → 15 min expiry\n\n"
                "<b>Best session:</b> London/NY Overlap (13:00–16:00 UTC)"
            )

        elif text.startswith("/help"):
            _send_message(chat_id,
                "📖 <b>Signal Bot Pro — Help</b>\n\n"
                "/start  — Welcome message\n"
                "/status — Bot health &amp; uptime\n"
                "/stats  — Today's win/loss report\n"
                "/pairs  — Assets &amp; timeframes\n"
                "/help   — This menu\n\n"
                "Signals are sent automatically when high-confidence "
                "setups are detected. You don't need to do anything — "
                "just wait for alerts."
            )

        else:
            # Ignore unknown commands silently (don't spam unknown users)
            if text.startswith("/"):
                _send_message(chat_id,
                    f"Unknown command: <code>{text}</code>\n"
                    "Type /help to see available commands."
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_start_time = datetime.utcnow()

def _get_uptime() -> str:
    delta = datetime.utcnow() - _start_time
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _ml_model_loaded() -> bool:
    try:
        from ai_model import MODEL_PATH
        import os
        return os.path.exists(MODEL_PATH)
    except Exception:
        return False

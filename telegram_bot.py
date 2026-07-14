"""
Telegram Bot — Sends signals to Free and VIP channels.
Supports inline trade execution via Deriv API for connected subscribers.

Commands:
  /start        — Welcome
  /status       — Bot status
  /stats        — Today's performance
  /pairs        — Monitored pairs
  /help         — Help menu
  /connect      — Link Deriv account
  /disconnect   — Unlink Deriv account
  /balance      — Check Deriv balance
  /setamount    — Set default trade amount
  /myaccount    — View account settings
  /connectpo    — Link Pocket Option account (OTC, self-service per-user)
  /disconnectpo — Unlink Pocket Option account
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WAT = timezone(timedelta(hours=1))

def _to_wat(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(WAT)

BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
FREE_CHANNEL  = os.getenv("TELEGRAM_FREE_CHANNEL", "")
VIP_CHANNEL   = os.getenv("TELEGRAM_VIP_CHANNEL", "")
ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# In-memory stores
_pending_trades: dict = {}   # signal_key → trade info
_awaiting_amount: dict = {}  # chat_id → signal_key (user typing custom amount)

# Pocket Option credential collection conversation state.
# chat_id → {"stage": "session" | "uid" | "mode", "session": str, "uid": str}
_awaiting_po_setup: dict = {}

# Persistent reply keyboard shown at bottom of every chat.
MAIN_MENU_KEYBOARD = {
    "keyboard": [
        [{"text": "📊 Pairs"},          {"text": "📈 Stats"},           {"text": "✅ Status"}],
        [{"text": "🔗 Connect Deriv"},  {"text": "💰 Balance"},         {"text": "❌ Disconnect"}],
        [{"text": "🎯 Connect OTC"},    {"text": "🔌 Disconnect OTC"}],
        [{"text": "💵 Set Amount"},     {"text": "👤 My Account"},      {"text": "📖 Help"}],
    ],
    "resize_keyboard": True,
    "persistent": True,
}


# ---------------------------------------------------------------------------
# Low-level senders
# ---------------------------------------------------------------------------

def _send_message(chat_id: str, text: str, parse_mode: str = "HTML",
                  reply_markup: dict = None) -> bool:
    if not BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured — message not sent.")
        return False

    url  = TELEGRAM_API.format(token=BOT_TOKEN, method="sendMessage")
    data = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    if reply_markup:
        data["reply_markup"] = reply_markup

    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.ok:
            logger.info(f"Telegram message sent to {chat_id}")
            return True
        else:
            logger.error(f"Telegram error: {resp.status_code} {resp.text[:500]}")
            if "can't parse" in resp.text:
                data["parse_mode"] = ""
                data.pop("reply_markup", None)
                resp2 = requests.post(url, json=data, timeout=10)
                if resp2.ok:
                    logger.info(f"Sent (plain text fallback) to {chat_id}")
                    return True
            return False
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")
        return False


def _answer_callback(callback_query_id: str, text: str = "", alert: bool = False):
    url  = TELEGRAM_API.format(token=BOT_TOKEN, method="answerCallbackQuery")
    data = {"callback_query_id": callback_query_id, "text": text, "show_alert": alert}
    try:
        requests.post(url, json=data, timeout=5)
    except Exception:
        pass


def _edit_message(chat_id: str, message_id: int, text: str, parse_mode: str = "HTML",
                  reply_markup: dict = None):
    url  = TELEGRAM_API.format(token=BOT_TOKEN, method="editMessageText")
    data = {"chat_id": chat_id, "message_id": message_id,
            "text": text, "parse_mode": parse_mode}
    if reply_markup:
        data["reply_markup"] = reply_markup
    try:
        requests.post(url, json=data, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Signal formatters
# ---------------------------------------------------------------------------

def _signal_times(signal, sent_at: datetime):
    """Return (entry_dt, expiry_dt, seconds_left) — all UTC-aware."""
    entry_dt  = signal.timestamp
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    expiry_dt = entry_dt + timedelta(minutes=signal.expiry_min)
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    secs_left = int((expiry_dt - sent_at).total_seconds())
    return entry_dt, expiry_dt, secs_left


def _format_timing_block(signal, sent_at: datetime) -> str:
    """
    Shows when signal was received, when to enter (candle close), 
    and when it expires.
    """
    entry_dt, expiry_dt, secs_left = _signal_times(signal, sent_at)

    entry_str  = _to_wat(entry_dt).strftime('%H:%M:%S')
    expiry_str = _to_wat(expiry_dt).strftime('%H:%M:%S')
    sent_str   = _to_wat(sent_at).strftime('%H:%M:%S')

    lag = int((sent_at.replace(tzinfo=timezone.utc)
               - entry_dt.replace(tzinfo=timezone.utc)).total_seconds())

    if secs_left <= 0:
        urgency = "⛔ <b>TOO LATE — do not enter this one</b>"
    elif lag <= 5:
        urgency = f"🚀 <b>ENTER NOW</b> — about {secs_left}s left"
    else:
        urgency = f"⚡ <b>ENTER IMMEDIATELY</b> — only ~{secs_left}s left"

    timing_lines = [
        f"⏰ <b>Signal received:</b> {sent_str} WAT",
        f"🎯 <b>ENTER at:</b>  <b>{entry_str} WAT</b> (candle close)",
        f"⌛ <b>Expires at:</b> <b>{expiry_str} WAT</b>  ({signal.expiry_min} min)",
        f"{urgency}"
    ]
    
    # Show how long until entry if signal came early
    if lag > 2 and secs_left > 0:
        timing_lines.insert(2, f"⏱️ <b>Wait:</b> {lag}s until entry time")
    
    return "\n".join(timing_lines) + "\n"


def _format_free_message(signal, sent_at: datetime = None) -> str:
    is_call  = signal.direction == "CALL"
    icon     = "🟢" if is_call else "🔴"
    label    = "CALL ↑ BUY" if is_call else "PUT ↓ SELL"
    action   = "Price expected to RISE" if is_call else "Price expected to FALL"
    sent_at  = sent_at or datetime.now(timezone.utc)

    timing_block = _format_timing_block(signal, sent_at)

    return (
        f"🔔 <b>New Signal</b>\n\n"
        f"{icon} <b>{label}</b> — {signal.asset}\n"
        f"📈 {action}\n"
        f"🤖 Confidence: <b>{signal.confidence:.0f}%</b>\n"
        f"{timing_block}"
        f"\n📊 Full analysis available in VIP 👇\n"
        f"🔒 Join: {os.getenv('VIP_INVITE_LINK', 't.me/your_vip_link')}"
    )


def _format_vip_message(signal, sent_at: datetime = None) -> str:
    is_call  = signal.direction == "CALL"
    icon     = "🟢" if is_call else "🔴"
    label    = "CALL ↑ BUY" if is_call else "PUT ↓ SELL"
    action   = "📈 Price expected to RISE — place a CALL/BUY trade" if is_call \
               else "📉 Price expected to FALL — place a PUT/SELL trade"
    sent_at  = sent_at or datetime.now(timezone.utc)

    timing_block = _format_timing_block(signal, sent_at)

    return (
        f"{'⭐' if is_call else '🔥'} <b>VIP Signal</b> — {icon} {label}\n\n"
        f"{action}\n\n"
        f"📊 <b>Pair:</b>       {signal.asset}\n"
        f"📈 <b>Timeframe:</b>  {signal.timeframe}\n"
        f"⏳ <b>Expiry:</b>     {signal.expiry_min} minutes\n"
        f"💰 <b>Entry price:</b> <code>{signal.entry_price:.5f}</code>\n"
        f"🤖 <b>Confidence:</b> <b>{signal.confidence:.0f}%</b>\n"
        f"🌍 <b>Session:</b>    {signal.session}\n\n"
        f"{timing_block}"
        f"⚠️ <i>Risk disclaimer: Binary options carry significant financial risk. "
        f"Never trade with money you cannot afford to lose. Past performance does not guarantee future results.</i>"
    )


def _format_trade_message_from_dict(trade: dict) -> str:
    is_call = trade["direction"] == "CALL"
    icon    = "🟢" if is_call else "🔴"
    label   = "CALL ↑" if is_call else "PUT ↓"
    return (
        f"{icon} <b>{label} — {trade['asset']}</b>\n"
        f"⏳ Expiry: <b>{trade['expiry_min']} min</b> (auto-set)\n"
        f"💰 Entry: <code>{trade['entry']:.5f}</code>\n"
        f"🤖 Confidence: <b>{trade['confidence']:.0f}%</b>\n\n"
        f"💵 <b>Amount: ${trade['amount']:.2f}</b>\n"
        f"Tap ➖ ➕ to adjust or ✏️ to type custom amount"
    )


# ---------------------------------------------------------------------------
# Trade keyboard
# ---------------------------------------------------------------------------

def _trade_keyboard(signal_key: str, amount: float, direction: str) -> dict:
    label = "✅ Place CALL" if direction == "CALL" else "✅ Place PUT"
    return {
        "inline_keyboard": [
            [
                {"text": "➖ $5",         "callback_data": f"amt_minus5_{signal_key}"},
                {"text": f"💵 ${amount:.0f}", "callback_data": f"amt_show_{signal_key}"},
                {"text": "➕ $5",         "callback_data": f"amt_plus5_{signal_key}"},
            ],
            [
                {"text": "➖ $1",         "callback_data": f"amt_minus1_{signal_key}"},
                {"text": "✏️ Type Amount","callback_data": f"amt_type_{signal_key}"},
                {"text": "➕ $1",         "callback_data": f"amt_plus1_{signal_key}"},
            ],
            [
                {"text": label,            "callback_data": f"trade_execute_{signal_key}"},
                {"text": "❌ Skip",        "callback_data": f"trade_skip_{signal_key}"},
            ],
        ]
    }


# ---------------------------------------------------------------------------
# Public send functions
# ---------------------------------------------------------------------------

def send_prealert(asset: str, timeframe: str, direction: str,
                  confidence: float, seconds_left: int) -> dict:
    """Broadcast a provisional heads-up that a signal is likely forming."""
    results = {}
    text = (
        f"⏰ <b>GET READY</b>\n\n"
        f"📊 Pair: <b>{asset}</b>\n"
        f"⏱️ Timeframe: <b>{timeframe}</b>\n"
        f"⏳ Signal may come in ~<b>{seconds_left}s</b> (next candle close)\n\n"
        f"👉 Open <b>{asset}</b> now and get ready. "
        f"Wait for the confirmed signal for direction."
    )

    tasks = []
    if FREE_CHANNEL:
        tasks.append(("free", FREE_CHANNEL))
    if VIP_CHANNEL:
        tasks.append(("vip", VIP_CHANNEL))

    def _send_task(key, chat_id):
        results[key] = _send_message(chat_id, text)

    threads = [
        threading.Thread(target=_send_task, args=(key, chat_id), daemon=True)
        for key, chat_id in tasks
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=8)

    return results


def send_signal(signal) -> dict:
    results = {}
    sent_at = datetime.now(timezone.utc)

    tasks = []
    if FREE_CHANNEL:
        tasks.append(("free", FREE_CHANNEL, _format_free_message(signal, sent_at)))
    if VIP_CHANNEL:
        tasks.append(("vip", VIP_CHANNEL, _format_vip_message(signal, sent_at)))

    def _send_task(key, chat_id, text):
        results[key] = _send_message(chat_id, text)

    threads = [
        threading.Thread(target=_send_task, args=(key, chat_id, text), daemon=True)
        for key, chat_id, text in tasks
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=12)

    if ADMIN_CHAT_ID:
        threading.Thread(
            target=_send_trade_prompt,
            args=(ADMIN_CHAT_ID, signal),
            daemon=True
        ).start()

    return results


# ---------------------------------------------------------------------------
# OTC (Pocket Option) — PRIVATE per-user delivery
# ---------------------------------------------------------------------------

def _otc_display_asset(asset: str) -> str:
    base = asset.replace("_otc", "").replace("_OTC", "").rstrip("_")
    return f"{base} OTC"


def _format_otc_message(signal, sent_at: datetime = None) -> str:
    is_call = signal.direction == "CALL"
    icon    = "🟢" if is_call else "🔴"
    label   = "CALL ↑ BUY" if is_call else "PUT ↓ SELL"
    action  = "📈 Price expected to RISE — place a CALL/BUY trade" if is_call \
              else "📉 Price expected to FALL — place a PUT/SELL trade"
    sent_at = sent_at or datetime.now(timezone.utc)

    timing_block = _format_timing_block(signal, sent_at)

    return (
        f"🎯 <b>OTC Signal</b> (Pocket Option) — {icon} {label}\n\n"
        f"{action}\n\n"
        f"📊 <b>Pair:</b>       {_otc_display_asset(signal.asset)}\n"
        f"📈 <b>Timeframe:</b>  {signal.timeframe}\n"
        f"⏳ <b>Expiry:</b>     {signal.expiry_min} minutes\n"
        f"💰 <b>Entry price:</b> <code>{signal.entry_price:.5f}</code>\n"
        f"🤖 <b>Confidence:</b> <b>{signal.confidence:.0f}%</b>\n"
        f"🌍 <b>Session:</b>    {signal.session}\n"
        f"\n{timing_block}"
        f"⚠️ <i>OTC markets are broker-generated. Binary options carry significant "
        f"financial risk. Never trade with money you cannot afford to lose.</i>"
    )


def _otc_asset_matches(user_assets, signal_asset: str) -> bool:
    if not user_assets:
        return True
    target = signal_asset.upper().replace("_OTC", "").rstrip("_")
    for a in user_assets:
        if a.upper().replace("_OTC", "").rstrip("_") == target:
            return True
    return False


def send_otc_signal(signal) -> dict:
    results: dict = {}
    sent_at = datetime.now(timezone.utc)

    try:
        from user_manager import UserManager
        po_users = UserManager().get_all_platform_users("pocket_option")
    except Exception as exc:
        logger.error(f"send_otc_signal: could not load Pocket Option users — {exc}")
        return results

    if not po_users:
        logger.info("send_otc_signal: no Pocket Option users connected")
        return results

    text = _format_otc_message(signal, sent_at)

    def _send_task(tid: str):
        results[tid] = _send_message(tid, text)

    threads = []
    for u in po_users:
        if not _otc_asset_matches(u.get("assets"), signal.asset):
            continue
        tid = str(u["telegram_id"])
        t = threading.Thread(target=_send_task, args=(tid,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=12)

    logger.info(
        f"send_otc_signal: {signal.asset}/{signal.timeframe} "
        f"{signal.direction} delivered privately to {len(results)} PO user(s)."
    )
    return results


def _send_trade_prompt(chat_id: str, signal):
    try:
        from user_manager import get_subscriber
        sub = get_subscriber(chat_id)
        if not sub or not sub.get("deriv_token"):
            return
        amount = sub.get("trade_amount", 10.0)
    except Exception:
        amount = 10.0

    import hashlib
    signal_key = hashlib.md5(
        f"{signal.asset}{signal.timeframe}{signal.direction}{signal.timestamp}".encode()
    ).hexdigest()[:8]

    _pending_trades[signal_key] = {
        "asset":      signal.asset,
        "direction":  signal.direction,
        "expiry_min": signal.expiry_min,
        "entry":      signal.entry_price,
        "confidence": signal.confidence,
        "amount":     amount,
        "chat_id":    chat_id,
    }

    text     = _format_trade_message_from_dict(_pending_trades[signal_key])
    keyboard = _trade_keyboard(signal_key, amount, signal.direction)
    _send_message(chat_id, text, reply_markup=keyboard)


def send_trade_prompts_to_subscribers(signal):
    try:
        from user_manager import get_all_connected
        for sub in get_all_connected():
            try:
                _send_trade_prompt(sub["chat_id"], signal)
            except Exception as exc:
                logger.error(f"Trade prompt error for {sub['chat_id']}: {exc}")
    except Exception as exc:
        logger.error(f"send_trade_prompts error: {exc}")


def send_admin_alert(text: str):
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, f"🤖 <b>Bot Alert</b>\n\n{text}")


def send_performance_report(report_text: str, report_type: str = "Daily"):
    msg = (
        f"📊 <b>{report_type} Performance Report</b>\n"
        f"🕒 {_to_wat(datetime.now(timezone.utc)).strftime('%Y-%m-%d %H:%M WAT')}\n\n"
        f"{report_text}"
    )
    if VIP_CHANNEL:
        _send_message(VIP_CHANNEL, msg)
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, msg)


# ---------------------------------------------------------------------------
# Callback query handler
# ---------------------------------------------------------------------------

def _handle_callback(update: dict):
    cb      = update.get("callback_query", {})
    if not cb:
        return

    cb_id   = cb["id"]
    data    = cb.get("data", "")
    chat_id = str(cb["from"]["id"])
    msg_id  = cb["message"]["message_id"]

    parts = data.split("_", 2)
    if len(parts) < 3:
        _answer_callback(cb_id)
        return

    action     = f"{parts[0]}_{parts[1]}"
    signal_key = parts[2]
    trade      = _pending_trades.get(signal_key)

    if not trade:
        _answer_callback(cb_id, "⏰ Signal expired", alert=True)
        return

    if action == "amt_minus5":
        trade["amount"] = max(1, trade["amount"] - 5)

    elif action == "amt_plus5":
        trade["amount"] = trade["amount"] + 5

    elif action == "amt_minus1":
        trade["amount"] = max(1, trade["amount"] - 1)

    elif action == "amt_plus1":
        trade["amount"] = trade["amount"] + 1

    elif action == "amt_show":
        _answer_callback(cb_id, f"Current amount: ${trade['amount']:.2f}")
        return

    elif action == "amt_type":
        _awaiting_amount[chat_id] = signal_key
        _answer_callback(cb_id)
        _send_message(chat_id,
            f"✏️ <b>Type your trade amount</b>\n\n"
            f"Send any number e.g. <code>25</code> or <code>150.50</code>\n"
            f"Minimum: $1\n\n"
            f"Current amount: <b>${trade['amount']:.2f}</b>"
        )
        return

    elif action == "trade_skip":
        _pending_trades.pop(signal_key, None)
        _awaiting_amount.pop(chat_id, None)
        _answer_callback(cb_id, "Skipped ✓")
        _edit_message(chat_id, msg_id,
            f"❌ <b>Trade Skipped</b>\n"
            f"{trade['asset']} {trade['direction']}")
        return

    elif action == "trade_execute":
        try:
            from user_manager import get_subscriber
            sub = get_subscriber(chat_id)
            if not sub or not sub.get("deriv_token"):
                _answer_callback(cb_id, "❌ Not connected. Use /connect first", alert=True)
                return

            _answer_callback(cb_id, "⏳ Placing trade...")

            from trade_executor import place_trade
            result = place_trade(
                token      = sub["deriv_token"],
                asset      = trade["asset"],
                direction  = trade["direction"],
                amount     = trade["amount"],
                expiry_min = trade["expiry_min"],
            )

            _pending_trades.pop(signal_key, None)
            _awaiting_amount.pop(chat_id, None)

            if result["success"]:
                payout   = result.get("payout", 0)
                profit   = round(payout - trade["amount"], 2)
                bal      = result.get("balance_after", "?")
                currency = result.get("currency", "USD")
                _edit_message(chat_id, msg_id,
                    f"✅ <b>Trade Placed!</b>\n\n"
                    f"📊 {trade['asset']} — {trade['direction']}\n"
                    f"⏳ Expiry: {trade['expiry_min']} min\n"
                    f"💵 Stake: ${trade['amount']:.2f}\n"
                    f"🏆 Potential payout: ${payout:.2f} (+${profit:.2f})\n"
                    f"💰 Balance after: {bal} {currency}\n\n"
                    f"⚠️ <i>Result known after expiry.</i>"
                )
                logger.info(
                    f"Trade placed: {chat_id} | {trade['asset']} "
                    f"{trade['direction']} ${trade['amount']}"
                )
            else:
                error = result.get("error", "Unknown error")
                _edit_message(chat_id, msg_id,
                    f"❌ <b>Trade Failed</b>\n\n"
                    f"Reason: {error}\n\n"
                    f"Check your Deriv account and try again."
                )
        except Exception as exc:
            logger.error(f"Trade execution error: {exc}")
            _answer_callback(cb_id, f"❌ Error: {exc}", alert=True)
        return

    text     = _format_trade_message_from_dict(trade)
    keyboard = _trade_keyboard(signal_key, trade["amount"], trade["direction"])
    _edit_message(chat_id, msg_id, text, reply_markup=keyboard)
    _answer_callback(cb_id)


# ---------------------------------------------------------------------------
# Pocket Option credential conversation
# ---------------------------------------------------------------------------

def _handle_po_setup_input(chat_id: str, text: str):
    state = _awaiting_po_setup.get(chat_id)
    if not state:
        return

    stage = state.get("stage")

    if stage == "session":
        state["session"] = text.strip()
        state["stage"] = "uid"
        _send_message(chat_id,
            "✅ Session received.\n\n"
            "Now send your Pocket Option <b>uid</b> (a number from the same "
            "auth payload):"
        )
        return

    if stage == "uid":
        uid_raw = text.strip()
        if not uid_raw.isdigit():
            _send_message(chat_id, "❌ uid should be numbers only. Try again:")
            return
        state["uid"] = uid_raw
        state["stage"] = "mode"
        _send_message(chat_id,
            "✅ uid received.\n\n"
            "Is this a <b>demo</b> or <b>real</b> account? Reply with "
            "<code>demo</code> or <code>real</code>:"
        )
        return

    if stage == "mode":
        mode = text.strip().lower()
        if mode not in ("demo", "real"):
            _send_message(chat_id, "❌ Please reply with <code>demo</code> or <code>real</code>:")
            return

        is_demo = mode == "demo"
        try:
            from user_manager import UserManager
            ok = UserManager().save_platform_credentials(
                telegram_id=chat_id,
                platform="pocket_option",
                credentials={"session": state["session"], "uid": state["uid"]},
                is_demo=is_demo,
            )
        except Exception as exc:
            ok = False
            logger.error(f"save_platform_credentials failed: {exc}")

        _awaiting_po_setup.pop(chat_id, None)

        if ok:
            _send_message(chat_id,
                f"✅ <b>Pocket Option connected!</b> ({'DEMO' if is_demo else 'REAL'})\n\n"
                f"You'll start receiving OTC signals sourced from your Pocket "
                f"Option feed on the next scan cycle.\n"
                f"Use /disconnectpo anytime to unlink."
            )
        else:
            _send_message(chat_id,
                "❌ Couldn't save your Pocket Option credentials. Please try "
                "/connectpo again."
            )
        return


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

class BotCommandHandler:
    def __init__(self):
        self.offset  = 0
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    def start(self):
        if not BOT_TOKEN:
            logger.warning("No TELEGRAM_BOT_TOKEN — command handler disabled.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="TelegramPoller",
            daemon=True,
        )
        self._thread.start()
        logger.info("Telegram command handler started (background thread).")

    def stop(self):
        self._stop.set()

    def poll_once(self):
        if self._thread and self._thread.is_alive():
            return
        self._do_poll()

    def _poll_loop(self):
        logger.info("Telegram poll loop running.")
        while not self._stop.is_set():
            try:
                self._do_poll()
            except Exception as exc:
                logger.error(f"Poll loop error: {exc}")
                time.sleep(5)

    def _do_poll(self):
        if not BOT_TOKEN:
            return
        url  = TELEGRAM_API.format(token=BOT_TOKEN, method="getUpdates")
        data = {"offset": self.offset, "timeout": 25, "limit": 10}
        try:
            resp = requests.get(url, params=data, timeout=30)
            if not resp.ok:
                logger.warning(f"getUpdates returned {resp.status_code}")
                time.sleep(3)
                return
            updates = resp.json().get("result", [])
            for update in updates:
                self.offset = update["update_id"] + 1
                try:
                    if "callback_query" in update:
                        _handle_callback(update)
                    else:
                        self._handle_update(update)
                except Exception as exc:
                    logger.error(f"Handle update error: {exc}")
        except requests.exceptions.Timeout:
            pass
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

        if chat_id in _awaiting_amount and not text.startswith("/"):
            try:
                amount = float(text.replace("$", "").strip())
                if amount < 1:
                    _send_message(chat_id, "❌ Minimum amount is $1. Try again:")
                    return

                signal_key = _awaiting_amount.pop(chat_id)

                if signal_key == "__setamount__":
                    from user_manager import set_amount
                    set_amount(chat_id, amount)
                    _send_message(chat_id,
                        f"✅ <b>Default amount set to ${amount:.2f}</b>\n\n"
                        f"You can still change it per signal using ✏️ Type Amount."
                    )
                    return

                trade = _pending_trades.get(signal_key)
                if not trade:
                    _send_message(chat_id, "⏰ Signal expired. Wait for the next signal.")
                    return
                trade["amount"] = amount
                text_msg  = _format_trade_message_from_dict(trade)
                keyboard  = _trade_keyboard(signal_key, amount, trade["direction"])
                _send_message(chat_id,
                    f"✅ Amount set to <b>${amount:.2f}</b>\n\n" + text_msg,
                    reply_markup=keyboard
                )
                return
            except ValueError:
                _send_message(chat_id,
                    "❌ Invalid amount. Send a number like <code>25</code> or <code>150.50</code>:"
                )
                return

        if chat_id in _awaiting_po_setup and not text.startswith("/"):
            _handle_po_setup_input(chat_id, text)
            return

        button_map = {
            "📊 Pairs":           "/pairs",
            "📈 Stats":           "/stats",
            "✅ Status":          "/status",
            "🔗 Connect Deriv":   "/connect",
            "💰 Balance":         "/balance",
            "❌ Disconnect":      "/disconnect",
            "🎯 Connect OTC":     "/connectpo",
            "🔌 Disconnect OTC":  "/disconnectpo",
            "💵 Set Amount":      "/setamount",
            "👤 My Account":      "/myaccount",
            "📖 Help":            "/help",
            "🔗 Connect":         "/connect",
            "🎯 Pocket Option":   "/connectpo",
        }
        if text in button_map:
            text = button_map[text]

        if text.startswith("/start"):
            first_name = msg.get("from", {}).get("first_name", "")
            try:
                from user_manager import UserManager
                UserManager().register_user_from_telegram(chat_id, username, first_name)
            except Exception as exc:
                logger.error(f"register_user_from_telegram failed: {exc}")

            _send_message(chat_id,
                "👋 Welcome to <b>Signal Bot Pro</b>!\n\n"
                "I generate high-confidence binary options signals using "
                "AI + market structure analysis.\n\n"
                "<b>Signal Types:</b>\n"
                "🟢 <b>CALL ↑</b> — Price expected to RISE\n"
                "🔴 <b>PUT ↓</b> — Price expected to FALL\n\n"
                "Use the menu buttons below to get started.\n"
                "Connect your Deriv account to execute trades directly from signals, "
                "or connect Pocket Option (OTC) with 🎯 Pocket Option.\n\n"
                "⚠️ <i>Trading involves significant risk. Never invest more than you can afford to lose.</i>",
                reply_markup=MAIN_MENU_KEYBOARD
            )

        elif text.startswith("/connectpo"):
            _awaiting_po_setup[chat_id] = {"stage": "session"}
            _send_message(chat_id,
                "🎯 <b>Connect Pocket Option (OTC)</b>\n\n"
                "This links your OWN Pocket Option account so you get "
                "signals sourced from Pocket Option's OTC feed.\n\n"
                "Steps:\n"
                "1. Log in to Pocket Option in your browser\n"
                "2. Open DevTools → Application/Storage → cookies (or the "
                "network tab) and copy your <b>session</b> value\n\n"
                "Send me your session value now:"
            )

        elif text.startswith("/disconnectpo"):
            try:
                from user_manager import UserManager
                UserManager().remove_platform_credentials(chat_id, "pocket_option")
                _awaiting_po_setup.pop(chat_id, None)
                _send_message(chat_id,
                    "✅ <b>Pocket Option disconnected.</b>\n\n"
                    "Use 🎯 Pocket Option anytime to reconnect."
                )
            except Exception as exc:
                _send_message(chat_id, f"❌ Error: {exc}")

        elif text.startswith("/connect"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                _send_message(chat_id,
                    "🔗 <b>Connect Your Deriv Account</b>\n\n"
                    "Steps:\n"
                    "1. Go to <b>app.deriv.com/account/api-token</b>\n"
                    "2. Create token with ✅ <b>Trade</b> permission only\n"
                    "3. Send your token:\n\n"
                    "<code>/connect YOUR_TOKEN_HERE</code>"
                )
                return

            token = parts[1].strip()
            _send_message(chat_id, "⏳ Verifying your Deriv token...")
            try:
                from trade_executor import get_balance
                result = get_balance(token)
                if result["success"]:
                    from user_manager import save_token
                    save_token(chat_id, token, username)
                    _send_message(chat_id,
                        f"✅ <b>Deriv Account Connected!</b>\n\n"
                        f"👤 Name: {result.get('name', 'N/A')}\n"
                        f"🆔 Login ID: {result.get('loginid', 'N/A')}\n"
                        f"💰 Balance: {result['balance']} {result['currency']}\n\n"
                        f"You can now execute trades directly from signal buttons!\n\n"
                        f"Use /setamount to set your default trade amount.\n"
                        f"You can also type any custom amount per signal."
                    )
                else:
                    _send_message(chat_id,
                        f"❌ <b>Connection Failed</b>\n\n"
                        f"Error: {result['error']}\n\n"
                        f"Make sure your token has Read Trade permissions."
                    )
            except Exception as exc:
                _send_message(chat_id, f"❌ Error: {exc}")

        elif text.startswith("/disconnect"):
            try:
                from user_manager import remove_token
                remove_token(chat_id)
                _send_message(chat_id,
                    "✅ <b>Deriv account disconnected.</b>\n\n"
                    "Use /connect anytime to reconnect."
                )
            except Exception as exc:
                _send_message(chat_id, f"❌ Error: {exc}")

        elif text.startswith("/balance"):
            try:
                from user_manager import get_subscriber
                sub = get_subscriber(chat_id)
                if not sub or not sub.get("deriv_token"):
                    _send_message(chat_id,
                        "❌ No Deriv account connected.\n"
                        "Use /connect to link your account."
                    )
                    return
                from trade_executor import get_balance
                result = get_balance(sub["deriv_token"])
                if result["success"]:
                    _send_message(chat_id,
                        f"💰 <b>Deriv Balance</b>\n\n"
                        f"Balance: <b>{result['balance']} {result['currency']}</b>\n"
                        f"Login ID: {result.get('loginid', 'N/A')}"
                    )
                else:
                    _send_message(chat_id, f"❌ Error: {result['error']}")
            except Exception as exc:
                _send_message(chat_id, f"❌ Error: {exc}")

        elif text.startswith("/setamount"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                _awaiting_amount[chat_id] = "__setamount__"
                _send_message(chat_id,
                    "💵 <b>Set Default Trade Amount</b>\n\n"
                    "Type your amount and send e.g. <code>10</code> or <code>25.50</code>\n\n"
                    "This sets your default per trade.\n"
                    "You can still change it per signal using ✏️ Type Amount."
                )
                return
            try:
                amount = float(parts[1].strip().replace("$", ""))
                if amount < 1:
                    _send_message(chat_id, "❌ Minimum amount is $1")
                    return
                from user_manager import set_amount
                set_amount(chat_id, amount)
                _send_message(chat_id,
                    f"✅ <b>Default amount set to ${amount:.2f}</b>\n\n"
                    f"You can still type any custom amount per signal using ✏️ Type Amount."
                )
            except ValueError:
                _send_message(chat_id,
                    "❌ Invalid amount. Example: <code>/setamount 25</code>"
                )

        elif text.startswith("/myaccount"):
            try:
                from user_manager import get_subscriber
                sub = get_subscriber(chat_id)
                if not sub:
                    _send_message(chat_id,
                        "No account found.\nUse /connect to link your Deriv account."
                    )
                    return
                connected    = "✅ Connected" if sub.get("deriv_token") else "❌ Not connected"
                connected_at = sub.get("connected_at")
                connected_str = _to_wat(connected_at).strftime('%Y-%m-%d %H:%M WAT') \
                    if connected_at else "N/A"

                po_line = "❌ Not connected"
                try:
                    from user_manager import UserManager
                    po = UserManager().get_platform_credentials(chat_id, "pocket_option")
                    if po and po.get("active"):
                        po_line = f"✅ Connected ({'DEMO' if po['is_demo'] else 'REAL'})"
                except Exception:
                    pass

                _send_message(chat_id,
                    f"👤 <b>My Account</b>\n\n"
                    f"Deriv: {connected}\n"
                    f"Connected: {connected_str}\n"
                    f"Default trade amount: <b>${sub['trade_amount']:.2f}</b>\n\n"
                    f"🎯 Pocket Option: {po_line}\n\n"
                    f"/balance     — Check live balance\n"
                    f"/setamount   — Change default amount\n"
                    f"/disconnect  — Unlink Deriv account\n"
                    f"/connectpo   — Connect Pocket Option\n"
                    f"/disconnectpo — Unlink Pocket Option",
                    reply_markup=MAIN_MENU_KEYBOARD
                )
            except Exception as exc:
                _send_message(chat_id, f"❌ Error: {exc}")

        elif text.startswith("/status"):
            uptime = _get_uptime()
            _send_message(chat_id,
                f"✅ <b>Bot Status: ONLINE</b>\n"
                f"🕒 {_to_wat(datetime.now(timezone.utc)).strftime('%Y-%m-%d %H:%M WAT')}\n"
                f"⏱️ Uptime: {uptime}\n\n"
                f"📡 <b>Scanning:</b>\n"
                f"  Pairs: EURUSD · GBPUSD · XAUUSD · USDJPY · BTCUSD\n"
                f"  Timeframes: M1 · M2 · M3 · M5 · M15\n"
                f"  Interval: every 60s\n\n"
                f"🟢 CALL = Price rising | 🔴 PUT = Price falling\n\n"
                f"🤖 AI mode: {'ML model' if _ml_model_loaded() else 'Heuristic (pre-training)'}\n"
                f"🔒 Confidence: M1/M2/M3=60-65% | M5/M15=80%",
                reply_markup=MAIN_MENU_KEYBOARD
            )

        elif text.startswith("/stats"):
            try:
                from performance_tracker import generate_daily_report
                report = generate_daily_report()
                _send_message(chat_id, report)
            except Exception:
                _send_message(chat_id,
                    "📊 <b>Today's Stats</b>\n\n"
                    "No completed signals yet today.\n"
                    "Stats update after signals expire."
                )

        elif text.startswith("/pairs"):
            _send_message(chat_id,
                "📊 <b>Monitored Pairs</b>\n\n"
                "🔵 <b>EURUSD</b> — Euro / US Dollar\n"
                "🔵 <b>GBPUSD</b> — British Pound / US Dollar\n"
                "🟡 <b>XAUUSD</b> — Gold / US Dollar\n"
                "🔵 <b>USDJPY</b> — US Dollar / Japanese Yen\n"
                "🟠 <b>BTCUSD</b> — Bitcoin / US Dollar\n\n"
                "<b>Timeframes &amp; Expiry:</b>\n"
                "  M1  → 1 min expiry\n"
                "  M2  → 2 min expiry\n"
                "  M3  → 3 min expiry\n"
                "  M5  → 5 min expiry\n"
                "  M15 → 15 min expiry\n\n"
                "<b>Best session:</b> London/NY Overlap (14:00–17:00 WAT)"
            )

        elif text.startswith("/help"):
            _send_message(chat_id,
                "📖 <b>Signal Bot Pro — Help</b>\n\n"
                "/connect     — Link Deriv account\n"
                "/disconnect  — Unlink Deriv account\n"
                "/balance     — Check Deriv balance\n"
                "/setamount   — Set default trade amount\n"
                "/myaccount   — View your settings\n"
                "/connectpo   — Connect Pocket Option (OTC)\n"
                "/disconnectpo — Unlink Pocket Option\n"
                "/status      — Bot health &amp; uptime\n"
                "/stats       — Today's win/loss report\n"
                "/pairs       — Assets &amp; timeframes\n"
                "/help        — This menu\n\n"
                "🟢 <b>CALL</b> = Price going UP\n"
                "🔴 <b>PUT</b>  = Price going DOWN\n\n"
                "<b>How to trade from bot:</b>\n"
                "1. Connect with /connect\n"
                "2. When signal arrives, tap ➖ ➕ or ✏️ to set amount\n"
                "3. Tap ✅ Place CALL/PUT to execute instantly",
                reply_markup=MAIN_MENU_KEYBOARD
            )

        else:
            if text.startswith("/"):
                _send_message(chat_id,
                    f"Unknown command: <code>{text}</code>\n"
                    "Type /help to see available commands."
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_start_time = datetime.now(timezone.utc)

def _get_uptime() -> str:
    delta = datetime.now(timezone.utc) - _start_time
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:   return f"{h}h {m}m"
    elif m > 0: return f"{m}m {s}s"
    return f"{s}s"


def _ml_model_loaded() -> bool:
    try:
        from ai_model import MODEL_PATH
        import os
        return os.path.exists(MODEL_PATH)
    except Exception:
        return False

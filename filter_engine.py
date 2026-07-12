"""
Filter Engine — Critical for win rate.
Blocks trades during: high-impact news, Asian dead zone, low volatility,
high spread, and ranging/uncertain market structure.
"""

import os
import logging
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    allowed:  bool = True
    reasons:  List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class NewsEvent:
    title:     str
    datetime:  datetime
    currency:  str
    impact:    str   # "High" | "Medium" | "Low"


# ---------------------------------------------------------------------------
# Session filter — asset aware
# ---------------------------------------------------------------------------

# Default dead hours for most forex pairs (European/American assets)
DEAD_HOURS_UTC = list(range(22, 24)) + list(range(0, 7))

# ---------------------------------------------------------------------------
# OTC (Pocket Option) awareness
#
# OTC instruments are stored with an `_otc` suffix (e.g. XAUUSD_otc) to keep
# their broker-generated price series separate from the real-market series.
# The filters below are keyed on real-market symbols, so without these helpers
# every OTC asset would silently miss its config and fall back to defaults —
# e.g. XAUUSD_otc would miss gold's wide-spread allowance and be over-filtered.
#
# Two rules define OTC behaviour:
#   * It quotes 24/7 (weekends included) — session/weekend gates don't apply.
#   * Its config (spread, currencies) should resolve via the BASE symbol.
# ---------------------------------------------------------------------------

def is_otc_asset(asset: str) -> bool:
    """True if this is a Pocket Option OTC instrument (e.g. 'XAUUSD_otc')."""
    return bool(asset) and asset.upper().endswith("_OTC")


def base_symbol(asset: str) -> str:
    """Strip the OTC suffix to get the real-market symbol used as a config key.
    'XAUUSD_otc' -> 'XAUUSD'.  Non-OTC assets are returned unchanged."""
    if is_otc_asset(asset):
        return asset[:-4].upper()
    return asset


# Asset-specific dead hours
# EURUSD — European asset, block full Asian dead zone
# GBPUSD — European asset, block full Asian dead zone
# XAUUSD — European/US commodity, block full Asian dead zone
# USDJPY — Tokyo session active 00:00-09:00 UTC, only block quietest hours 03:00-06:00
# BTCUSD — crypto trades 24/7, never block
ASSET_DEAD_HOURS = {
    "EURUSD": list(range(22, 24)) + list(range(0, 7)),   # original dead zone 22:00-07:00
    "GBPUSD": list(range(22, 24)) + list(range(0, 7)),   # original dead zone 22:00-07:00
    "XAUUSD": list(range(22, 24)) + list(range(0, 7)),   # original dead zone 22:00-07:00
    "USDJPY": list(range(3, 6)),                          # only block 03:00-06:00, Tokyo active rest
    "BTCUSD": [],                                         # crypto trades 24/7
}

TRADING_WINDOWS = {
    "london":   (7,  16),
    "new_york": (13, 22),
    "overlap":  (13, 16),
}


def get_current_session(dt: Optional[datetime] = None, asset: str = "") -> str:
    dt = dt or datetime.utcnow()
    h  = dt.hour

    # OTC instruments quote 24/7 and don't follow real-market sessions. Report
    # the clock context without implying real-market liquidity.
    if is_otc_asset(asset):
        if 13 <= h < 16:
            return "OTC (Peak Hours)"
        if 7 <= h < 22:
            return "OTC (Active)"
        return "OTC (24/7)"

    if asset == "USDJPY":
        if 0 <= h < 9:
            return "Tokyo"
        if 7 <= h < 13:
            return "London"
        if 13 <= h < 16:
            return "London/NY Overlap"
        if 16 <= h < 22:
            return "New York"
        return "After Hours"

    if asset == "BTCUSD":
        if 13 <= h < 16:
            return "London/NY Overlap (Peak)"
        if 7 <= h < 16:
            return "London"
        if 16 <= h < 22:
            return "New York"
        return "Off-Peak (Crypto Active)"

    # Default for EURUSD, GBPUSD, XAUUSD
    if 13 <= h < 16:
        return "London/NY Overlap"
    if 7 <= h < 13:
        return "London"
    if 16 <= h < 22:
        return "New York"
    if 1 <= h < 7:
        return "Early Asian (Low Liquidity)"
    return "Asian/Dead"


def is_dead_session(dt: Optional[datetime] = None, asset: str = "") -> bool:
    dt = dt or datetime.utcnow()
    h  = dt.hour

    # OTC (Pocket Option) instruments are broker-generated and quote 24/7,
    # including weekends. Real-market session dead zones do not apply to them,
    # so never block an OTC asset on session hours.
    if is_otc_asset(asset):
        return False

    # Use asset-specific dead hours if available
    if asset and asset in ASSET_DEAD_HOURS:
        return h in ASSET_DEAD_HOURS[asset]

    # Default dead hours for unknown assets
    return h in DEAD_HOURS_UTC


def is_weekend(dt: Optional[datetime] = None) -> bool:
    dt = dt or datetime.utcnow()
    return dt.weekday() >= 5


# ---------------------------------------------------------------------------
# Volatility thresholds — timeframe-aware
# ---------------------------------------------------------------------------

ATR_THRESHOLD = {
    "M1":  0.015,
    "M2":  0.012,
    "M3":  0.012,
    "M5":  0.03,
    "M15": 0.03,
}

DEFAULT_ATR_THRESHOLD = 0.03


def is_low_volatility(atr_pct: float, timeframe: str = "M5") -> bool:
    threshold = ATR_THRESHOLD.get(timeframe, DEFAULT_ATR_THRESHOLD)
    return atr_pct < threshold


# ---------------------------------------------------------------------------
# News filter
# ---------------------------------------------------------------------------

MANUAL_HIGH_IMPACT = []

_news_cache: List[NewsEvent] = []
_news_cache_ts: Optional[datetime] = None
NEWS_CACHE_TTL_MIN = 60


def _fetch_news_events() -> List[NewsEvent]:
    events = []
    try:
        api_key = os.getenv("MARKETAUX_KEY", "")
        if api_key:
            url = (
                f"https://api.marketaux.com/v1/news/all?"
                f"api_token={api_key}&filter_entities=true&language=en&limit=20"
            )
            resp = requests.get(url, timeout=8)
            if resp.ok:
                for item in resp.json().get("data", []):
                    events.append(NewsEvent(
                        title=item.get("title", ""),
                        datetime=datetime.fromisoformat(item.get("published_at", "").replace("Z", "+00:00")),
                        currency="",
                        impact="High",
                    ))
                return events
    except Exception as exc:
        logger.debug(f"News fetch failed: {exc}")

    now = datetime.utcnow()
    first_friday = _first_weekday_of_month(now.year, now.month, 4)
    nfp_dt = datetime(now.year, now.month, first_friday, 13, 30)
    events.append(NewsEvent("NFP", nfp_dt, "USD", "High"))

    cpi_day = _nth_weekday(now.year, now.month, 2, 1)
    events.append(NewsEvent("CPI", datetime(now.year, now.month, cpi_day, 13, 30), "USD", "High"))

    return events


def _first_weekday_of_month(year: int, month: int, weekday: int) -> int:
    import calendar
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[weekday] != 0:
            return week[weekday]
    return 1


def _nth_weekday(year: int, month: int, n: int, weekday: int) -> int:
    count = 0
    for day in range(1, 32):
        try:
            dt = datetime(year, month, day)
            if dt.weekday() == weekday:
                count += 1
                if count == n:
                    return day
        except ValueError:
            break
    return 1


def get_news_events() -> List[NewsEvent]:
    global _news_cache, _news_cache_ts
    now = datetime.utcnow()
    if _news_cache_ts is None or (now - _news_cache_ts).seconds > NEWS_CACHE_TTL_MIN * 60:
        _news_cache    = _fetch_news_events()
        _news_cache_ts = now
    return _news_cache


def is_near_news(asset: str, dt: Optional[datetime] = None, window_min: int = 30) -> Tuple[bool, str]:
    dt     = dt or datetime.utcnow()
    events = get_news_events()
    asset_currencies = set()

    # Resolve the real-market symbol first: 'XAUUSD_otc' is 10 chars, so
    # without this the length check below would silently match nothing and
    # no news filtering would be applied at all.
    sym = base_symbol(asset)

    if len(sym) == 6:
        asset_currencies = {sym[:3], sym[3:]}
    elif sym == "XAUUSD":
        asset_currencies = {"USD", "XAU"}

    window = timedelta(minutes=window_min)

    for ev in events:
        if ev.impact != "High":
            continue
        if ev.currency and ev.currency not in asset_currencies and asset_currencies:
            continue
        ev_dt  = ev.datetime.replace(tzinfo=None) if ev.datetime.tzinfo else ev.datetime
        dt_cmp = dt.replace(tzinfo=None) if dt.tzinfo else dt
        if abs(ev_dt - dt_cmp) <= window:
            return True, ev.title

    return False, ""


# ---------------------------------------------------------------------------
# Spread filter
# ---------------------------------------------------------------------------

def is_high_spread(asset: str, spread_pct: float) -> bool:
    MAX_SPREAD = {
        "EURUSD": 0.01,
        "GBPUSD": 0.015,
        "XAUUSD": 0.05,
        "USDJPY": 0.01,
        "BTCUSD": 0.50,
    }
    # Resolve via the BASE symbol so OTC variants inherit the right threshold.
    # Without this, XAUUSD_otc would miss gold's 0.05 allowance and fall back
    # to the 0.02 default — wrongly rejecting most gold OTC signals as
    # "high spread", since gold's spread is naturally wide.
    return spread_pct > MAX_SPREAD.get(base_symbol(asset), 0.02)


# ---------------------------------------------------------------------------
# Main filter gate
# ---------------------------------------------------------------------------

def apply_filters(
    asset: str,
    timeframe: str,
    atr_pct: float,
    volatility_state: str,
    trend: str,
    dt: Optional[datetime] = None,
    spread_pct: float = 0.0,
) -> FilterResult:
    result = FilterResult()
    dt     = dt or datetime.utcnow()
    otc    = is_otc_asset(asset)

    # 1. Weekend — real markets only.
    #    OTC (Pocket Option) instruments are broker-generated and quote through
    #    the weekend; blocking them here would disable OTC signals on exactly
    #    the days OTC is most useful. Real-market assets are gated as before.
    if not otc and is_weekend(dt):
        result.allowed = False
        result.reasons.append("Weekend — markets closed")
        return result

    # 2. Dead session — asset aware
    if is_dead_session(dt, asset):
        result.allowed = False
        result.reasons.append(
            f"Dead session hour ({dt.hour}:00 UTC) — "
            f"{'Asian dead zone' if not asset else f'{asset} inactive at this hour'}"
        )
        return result

    # 3. News risk
    near_news, news_title = is_near_news(asset, dt)
    if near_news:
        result.allowed = False
        result.reasons.append(f"High-impact news risk: {news_title} (±30 min window)")
        return result

    # 4. Low volatility — timeframe aware
    if is_low_volatility(atr_pct, timeframe):
        threshold = ATR_THRESHOLD.get(timeframe, DEFAULT_ATR_THRESHOLD)
        result.allowed = False
        result.reasons.append(
            f"Low volatility (ATR% = {atr_pct:.4f}% < {threshold:.3f}% threshold for {timeframe}) — market inactive"
        )
        return result

    # 5. Ranging structure
    if trend == "ranging":
        result.allowed = False
        result.reasons.append("Market is ranging — no clear directional bias")
        return result

    # 6. Spread
    if spread_pct > 0 and is_high_spread(asset, spread_pct):
        result.allowed = False
        result.reasons.append(f"High spread: {spread_pct:.4f}% exceeds threshold for {asset}")
        return result

    # 7. Warnings (non-blocking)
    session = get_current_session(dt, asset)
    if "Overlap" in session:
        result.warnings.append(f"{session} — highest liquidity ✓")
    elif session == "New York" and dt.hour > 19:
        result.warnings.append("Late NY session — liquidity decreasing")
    elif session in ("Early Asian (Low Liquidity)", "After Hours"):
        result.warnings.append(f"{session} — lower volume, trade carefully")
    elif session == "Tokyo":
        result.warnings.append("Tokyo session — best for USDJPY signals")
    elif session == "Off-Peak (Crypto Active)":
        result.warnings.append("Off-peak hours — crypto still active but lower volume")

    if volatility_state == "high":
        result.warnings.append("High volatility — widen mental stop, reduce size")

    return result

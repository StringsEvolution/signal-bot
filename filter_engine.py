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
# Session filter
# ---------------------------------------------------------------------------

DEAD_HOURS_UTC = list(range(22, 24)) + list(range(0, 7))

TRADING_WINDOWS = {
    "london":   (7,  16),
    "new_york": (13, 22),
    "overlap":  (13, 16),
}


def get_current_session(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.utcnow()
    h  = dt.hour
    if 13 <= h < 16:
        return "London/NY Overlap"
    if 7 <= h < 13:
        return "London"
    if 16 <= h < 22:
        return "New York"
    return "Asian/Dead"


def is_dead_session(dt: Optional[datetime] = None) -> bool:
    dt = dt or datetime.utcnow()
    return dt.hour in DEAD_HOURS_UTC


def is_weekend(dt: Optional[datetime] = None) -> bool:
    dt = dt or datetime.utcnow()
    return dt.weekday() >= 5


# ---------------------------------------------------------------------------
# Volatility thresholds — timeframe-aware
# ---------------------------------------------------------------------------

# Shorter timeframes have smaller candles so ATR% is naturally lower.
# Each value is the MINIMUM ATR% required to pass the volatility filter.
ATR_THRESHOLD = {
    "M1":  0.015,   # was 0.03 for all — M1 candles are tiny, lower threshold
    "M2":  0.012,   # M2/M3 even smaller candles, needs lower threshold
    "M3":  0.012,
    "M5":  0.03,    # original threshold — unchanged
    "M15": 0.03,    # original threshold — unchanged
}

DEFAULT_ATR_THRESHOLD = 0.03   # fallback for any unrecognised timeframe


def is_low_volatility(atr_pct: float, timeframe: str = "M5") -> bool:
    """ATR below timeframe-specific threshold = dead market."""
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

    if len(asset) == 6:
        asset_currencies = {asset[:3], asset[3:]}
    elif asset == "XAUUSD":
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
    }
    return spread_pct > MAX_SPREAD.get(asset, 0.02)


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

    # 1. Weekend
    if is_weekend(dt):
        result.allowed = False
        result.reasons.append("Weekend — markets closed")
        return result

    # 2. Dead session
    if is_dead_session(dt):
        result.allowed = False
        result.reasons.append(f"Dead session hour ({dt.hour}:00 UTC) — Asian dead zone")
        return result

    # 3. News risk
    near_news, news_title = is_near_news(asset, dt)
    if near_news:
        result.allowed = False
        result.reasons.append(f"High-impact news risk: {news_title} (±30 min window)")
        return result

    # 4. Low volatility — now timeframe-aware
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
    session = get_current_session(dt)
    if session == "London/NY Overlap":
        result.warnings.append("London/NY Overlap — highest liquidity ✓")
    elif session == "New York" and dt.hour > 19:
        result.warnings.append("Late NY session — liquidity decreasing")

    if volatility_state == "high":
        result.warnings.append("High volatility — widen mental stop, reduce size")

    return result

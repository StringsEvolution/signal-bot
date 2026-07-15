"""
Market Structure Engine
Detects trend direction, support/resistance zones, breakouts, and pullbacks.
This is the primary gate — no signal proceeds without structure confirmation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class StructureResult:
    trend: str                      # "bullish" | "bearish" | "ranging"
    trend_strength: float           # 0–1
    support_zones: List[float]      = field(default_factory=list)
    resistance_zones: List[float]   = field(default_factory=list)
    at_support: bool                = False
    at_resistance: bool             = False
    breakout: Optional[str]         = None   # "bullish_break" | "bearish_break" | None
    fakeout: bool                   = False
    pullback: bool                  = False
    nearest_support: Optional[float]  = None
    nearest_resistance: Optional[float] = None
    sr_distance_pct: float          = 999.0  # % distance to nearest S/R level
    structure_valid: bool           = False  # gate: True only if clear structure


# ---------------------------------------------------------------------------
# Swing-high / swing-low detection
# ---------------------------------------------------------------------------

def _find_swings(df: pd.DataFrame, left: int = 5, right: int = 5) -> Tuple[List[float], List[float]]:
    """Return lists of swing-high and swing-low price levels."""
    highs, lows = [], []
    n = len(df)

    for i in range(left, n - right):
        window_h = df["high"].iloc[i - left: i + right + 1]
        window_l = df["low"].iloc[i - left: i + right + 1]

        if df["high"].iloc[i] == window_h.max():
            highs.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] == window_l.min():
            lows.append(float(df["low"].iloc[i]))

    return highs, lows


def _cluster_levels(levels: List[float], tolerance_pct: float = 0.002) -> List[float]:
    """Merge price levels that are within tolerance into a single zone."""
    if not levels:
        return []

    sorted_lvls = sorted(levels)
    clusters = []
    current = [sorted_lvls[0]]

    for lvl in sorted_lvls[1:]:
        if abs(lvl - current[-1]) / current[-1] <= tolerance_pct:
            current.append(lvl)
        else:
            clusters.append(float(np.mean(current)))
            current = [lvl]

    clusters.append(float(np.mean(current)))
    return clusters


# ---------------------------------------------------------------------------
# Trend detection — Higher-Highs/Higher-Lows method + price vs EMA
# ---------------------------------------------------------------------------

def _detect_trend(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Classify trend using:
    1. Swing HH/HL (bullish) or LH/LL (bearish).
    2. Price position relative to EMA50.
    Returns (trend, strength 0–1).
    """
    if len(df) < 20:
        return "ranging", 0.0

    closes = df["close"].values
    ema50  = _ema(closes, 50)

    # --- price vs EMA50
    price_above = closes[-1] > ema50[-1]
    price_below = closes[-1] < ema50[-1]

    # --- recent swing analysis (last 30 candles)
    recent = df.tail(30).reset_index(drop=True)
    swing_h, swing_l = _find_swings(recent, left=3, right=3)

    bull_signals = 0
    bear_signals = 0

    # EMA alignment
    if price_above:
        bull_signals += 2
    elif price_below:
        bear_signals += 2

    # EMA slope
    if len(ema50) >= 5:
        slope = ema50[-1] - ema50[-5]
        if slope > 0:
            bull_signals += 1
        elif slope < 0:
            bear_signals += 1

    # Swing structure — require at least 2 swings
    if len(swing_h) >= 2:
        if swing_h[-1] > swing_h[-2]:
            bull_signals += 2   # Higher high
        else:
            bear_signals += 2   # Lower high

    if len(swing_l) >= 2:
        if swing_l[-1] > swing_l[-2]:
            bull_signals += 2   # Higher low
        else:
            bear_signals += 2   # Lower low

    total = bull_signals + bear_signals
    if total == 0:
        return "ranging", 0.0

    bull_ratio = bull_signals / total
    bear_ratio = bear_signals / total

    THRESHOLD = 0.60
    # Require EMA and swing to agree — if they conflict, call it ranging
    ema_says_bull = price_above and (len(ema50) >= 5 and ema50[-1] > ema50[-5])
    ema_says_bear = price_below and (len(ema50) >= 5 and ema50[-1] < ema50[-5])

    if bull_ratio >= THRESHOLD:
        # Extra guard: if EMA is clearly bearish, downgrade to ranging
        if ema_says_bear and bear_signals >= 2:
            return "ranging", round(bull_ratio * 0.8, 2)
        strength = round(bull_ratio, 2)
        return "bullish", strength
    elif bear_ratio >= THRESHOLD:
        # Extra guard: if EMA is clearly bullish, downgrade to ranging
        if ema_says_bull and bull_signals >= 2:
            return "ranging", round(bear_ratio * 0.8, 2)
        strength = round(bear_ratio, 2)
        return "bearish", strength
    else:
        return "ranging", round(max(bull_ratio, bear_ratio), 2)


# ---------------------------------------------------------------------------
# Breakout / fakeout detection
# ---------------------------------------------------------------------------

def _detect_breakout(df: pd.DataFrame, resistance: List[float], support: List[float]) -> Tuple[Optional[str], bool]:
    """
    Check if the last candles broke through a key level.
    TRUE Fakeout (strict definition to avoid false positives on real data):
      - Price closed CLEARLY beyond a level (by more than 0.05% buffer)
      - Then closed CLEARLY back inside within 2 bars
    A single candle touching a level is NOT a fakeout.
    """
    if len(df) < 5 or (not resistance and not support):
        return None, False

    recent       = df.tail(5)
    latest_close = float(recent["close"].iloc[-1])
    prev_close   = float(recent["close"].iloc[-2])
    prev2_close  = float(recent["close"].iloc[-3])

    breakout_type = None
    fakeout       = False

    for r in resistance:
        # Breakout: prev bar closed below, current bar closes above
        if prev_close < r <= latest_close:
            breakout_type = "bullish_break"
        # Fakeout: prev2 closed clearly ABOVE resistance (by >0.05%)
        # AND current close is clearly BACK BELOW resistance (by >0.05%)
        buffer = r * 0.0005
        if prev2_close > r + buffer and latest_close < r - buffer:
            fakeout = True

    for s in support:
        # Breakdown: prev bar closed above, current bar closes below
        if prev_close > s >= latest_close:
            breakout_type = "bearish_break"
        # Fakeout: prev2 closed clearly BELOW support (by >0.05%)
        # AND current close is clearly BACK ABOVE support (by >0.05%)
        buffer = s * 0.0005
        if prev2_close < s - buffer and latest_close > s + buffer:
            fakeout = True

    return breakout_type, fakeout


def _detect_pullback(df: pd.DataFrame, trend: str, support: List[float], resistance: List[float]) -> bool:
    """
    Detect a pullback into a key zone in the direction opposite to trend.
    Bullish trend + price drops to support = pullback opportunity.
    """
    if len(df) < 10:
        return False

    latest_close = float(df["close"].iloc[-1])
    recent_high  = float(df["high"].tail(5).max())
    recent_low   = float(df["low"].tail(5).min())

    if trend == "bullish" and support:
        nearest_sup = min(support, key=lambda x: abs(x - latest_close))
        pct_away = abs(latest_close - nearest_sup) / latest_close
        # Price pulled back within 0.3% of support
        if pct_away < 0.003 and recent_low < nearest_sup * 1.001:
            return True

    elif trend == "bearish" and resistance:
        nearest_res = min(resistance, key=lambda x: abs(x - latest_close))
        pct_away = abs(latest_close - nearest_res) / latest_close
        if pct_away < 0.003 and recent_high > nearest_res * 0.999:
            return True

    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyse_structure(df: pd.DataFrame) -> StructureResult:
    """
    Full market structure analysis.
    Returns StructureResult — the primary gate for signal generation.
    """
    if len(df) < 50:
        return StructureResult(trend="ranging", trend_strength=0.0)

    # 1. Trend
    trend, strength = _detect_trend(df)
    # Pre-compute EMA50/200 for validity gate
    closes_arr = df["close"].values.astype(float)
    ema50_arr  = _ema(closes_arr, 50)
    ema200_arr = _ema(closes_arr, 200)

    # 2. S/R zones from swing highs/lows (full history, clustered)
    sh, sl = _find_swings(df, left=5, right=5)
    # Full clustered level lists — kept for breakout detection, which needs
    # levels on BOTH sides of price (a bullish break crosses a former
    # resistance that is now at/below price). Filtering to r>price / s<price
    # here would make _detect_breakout mathematically unable to ever fire.
    all_resistance = sorted(_cluster_levels(sh), reverse=True)
    all_support    = sorted(_cluster_levels(sl),  reverse=False)

    # Keep only closest 5 each side to reduce noise (used for at_support/
    # at_resistance, pullback, nearest-level distance — NOT breakout).
    price = float(df["close"].iloc[-1])
    resistance_zones = sorted([r for r in all_resistance if r > price])[:5]
    support_zones    = sorted([s for s in all_support    if s < price], reverse=True)[:5]

    # 3. Nearest levels and distance
    nearest_sup = support_zones[0]    if support_zones    else None
    nearest_res = resistance_zones[0] if resistance_zones else None

    distances = []
    if nearest_sup:
        distances.append(abs(price - nearest_sup) / price * 100)
    if nearest_res:
        distances.append(abs(price - nearest_res) / price * 100)
    sr_distance = round(min(distances), 4) if distances else 999.0

    # 4. At S/R — within 0.15%
    at_support    = nearest_sup is not None and abs(price - nearest_sup) / price < 0.0015
    at_resistance = nearest_res is not None and abs(price - nearest_res) / price < 0.0015

    # 5. Breakout / fakeout — use FULL level lists so a level that price has
    #    just crossed (now sitting at/below for a bull break, at/above for a
    #    bear break) is still available to match against.
    breakout, fakeout = _detect_breakout(df, all_resistance, all_support)

    # 6. Pullback — moved below so it uses effective_trend (resolved after EMA check)

    # 7. Structure validity gate
    # Valid = clear trend (or EMA-confirmed trend) + price at a meaningful level
    # Note: trend can be 'ranging' if swing and EMA disagree during a pullback;
    # in that case we use EMA direction as tiebreaker
    ema50_curr  = ema50_arr[-1]  if not np.isnan(ema50_arr[-1])  else 0
    ema200_curr = ema200_arr[-1] if not np.isnan(ema200_arr[-1]) else 0
    ema_bullish = ema50_curr > ema200_curr * 1.001
    ema_bearish = ema50_curr < ema200_curr * 0.999

    effective_trend = trend
    # If swing direction conflicts with EMA direction, trust EMA (price is truth)
    if trend == "ranging":
        if ema_bullish:
            effective_trend = "bullish"
        elif ema_bearish:
            effective_trend = "bearish"
    elif trend == "bearish" and ema_bullish:
        # Swing says bearish but EMA50 > EMA200 = likely a pullback in bullish trend
        effective_trend = "bullish"
        strength = round(strength * 0.8, 2)   # reduce strength slightly for conflict
    elif trend == "bullish" and ema_bearish:
        # Swing says bullish but EMA50 < EMA200 = likely a pullback in bearish trend
        effective_trend = "bearish"
        strength = round(strength * 0.8, 2)

    # Detect pullback using the effective (EMA-corrected) trend direction
    pullback = _detect_pullback(df, effective_trend, support_zones, resistance_zones)

    # Align at_support/at_resistance with effective trend
    trend_at_level = (
        (effective_trend == "bullish" and at_support) or
        (effective_trend == "bearish" and at_resistance) or
        pullback or
        breakout is not None
    )

    structure_valid = (
        effective_trend != "ranging"
        and strength >= 0.55
        and trend_at_level
        and not fakeout
    )

    # Update trend to effective_trend for downstream use
    trend = effective_trend

    return StructureResult(
        trend=trend,
        trend_strength=strength,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        at_support=at_support,
        at_resistance=at_resistance,
        breakout=breakout,
        fakeout=fakeout,
        pullback=pullback,
        nearest_support=nearest_sup,
        nearest_resistance=nearest_res,
        sr_distance_pct=sr_distance,
        structure_valid=structure_valid,
    )


# ---------------------------------------------------------------------------
# Utility: EMA (used internally)
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    n = len(series)
    result = np.zeros_like(series, dtype=float)
    if n == 0:
        return result
    k = 2 / (period + 1)
    # Warmup index is normally period-1. But OTC / live streams may hand us
    # FEWER than `period` candles (e.g. 86 rows with an EMA200 request), which
    # made `result[period-1]` an out-of-bounds write and crashed the handler.
    # Clamp the seed to the last available index and seed with the SMA of
    # whatever history we have. When n >= period this is byte-for-byte the old
    # behaviour; when n < period, EMA(period) degrades gracefully to "mean of
    # available closes", which is a sensible approximation for the trend gate.
    seed_idx = min(period, n) - 1
    result[seed_idx] = np.mean(series[:seed_idx + 1])
    for i in range(seed_idx + 1, n):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result

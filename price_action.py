"""
Price Action Module
Detects: Bullish/Bearish Engulfing, Pin Bars, Doji, Rejection candles.
Patterns must be at key S/R zones to be meaningful.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CandlePattern:
    name: str
    direction: str          # "bullish" | "bearish" | "neutral"
    strength: float         # 0–1
    description: str


@dataclass
class PriceActionResult:
    patterns: List[CandlePattern] = field(default_factory=list)
    bullish_bias: float = 0.0      # 0–1 composite bullish signal
    bearish_bias: float = 0.0      # 0–1 composite bearish signal
    dominant_pattern: Optional[CandlePattern] = None
    pattern_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candle geometry helpers
# ---------------------------------------------------------------------------

def _body(candle: pd.Series) -> float:
    return abs(candle["close"] - candle["open"])

def _upper_wick(candle: pd.Series) -> float:
    return candle["high"] - max(candle["open"], candle["close"])

def _lower_wick(candle: pd.Series) -> float:
    return min(candle["open"], candle["close"]) - candle["low"]

def _total_range(candle: pd.Series) -> float:
    return candle["high"] - candle["low"]

def _is_bullish(candle: pd.Series) -> bool:
    return candle["close"] > candle["open"]

def _is_bearish(candle: pd.Series) -> bool:
    return candle["close"] < candle["open"]


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------

def _check_engulfing(curr: pd.Series, prev: pd.Series) -> Optional[CandlePattern]:
    """
    Bullish engulfing: prev bearish, curr bullish body completely engulfs prev body.
    Bearish engulfing: prev bullish, curr bearish body completely engulfs prev body.
    """
    curr_body = _body(curr)
    prev_body = _body(prev)

    if prev_body == 0:
        return None

    if (_is_bearish(prev) and _is_bullish(curr)
            and curr["open"] < prev["close"]
            and curr["close"] > prev["open"]
            and curr_body > prev_body * 0.7):
        strength = min(curr_body / prev_body, 1.0) if prev_body > 0 else 0.5
        return CandlePattern("Bullish Engulfing", "bullish", round(strength, 2),
                              "Bearish candle followed by larger bullish candle")

    if (_is_bullish(prev) and _is_bearish(curr)
            and curr["open"] > prev["close"]
            and curr["close"] < prev["open"]
            and curr_body > prev_body * 0.7):
        strength = min(curr_body / prev_body, 1.0) if prev_body > 0 else 0.5
        return CandlePattern("Bearish Engulfing", "bearish", round(strength, 2),
                              "Bullish candle followed by larger bearish candle")

    return None


def _check_pin_bar(candle: pd.Series) -> Optional[CandlePattern]:
    """
    Pin bar: small body, long wick on one side (≥2× body), tiny wick on other side.
    Bullish pin = long lower wick (hammer/rejection of low).
    Bearish pin = long upper wick (shooting star/rejection of high).
    """
    total = _total_range(candle)
    if total == 0:
        return None

    body  = _body(candle)
    upper = _upper_wick(candle)
    lower = _lower_wick(candle)

    # Body should be ≤ 35% of range; one wick ≥ 60% of range
    body_ratio  = body  / total
    upper_ratio = upper / total
    lower_ratio = lower / total

    if body_ratio > 0.45:
        return None

    if lower_ratio >= 0.55 and upper_ratio <= 0.25:
        strength = round(lower_ratio, 2)
        return CandlePattern("Bullish Pin Bar", "bullish", strength,
                              f"Long lower wick ({lower_ratio:.0%}) rejecting bearish pressure")

    if upper_ratio >= 0.55 and lower_ratio <= 0.25:
        strength = round(upper_ratio, 2)
        return CandlePattern("Bearish Pin Bar", "bearish", strength,
                              f"Long upper wick ({upper_ratio:.0%}) rejecting bullish pressure")

    return None


def _check_doji(candle: pd.Series) -> Optional[CandlePattern]:
    """
    Doji: body ≤ 10% of total range — indecision, neutral.
    """
    total = _total_range(candle)
    if total == 0:
        return None

    body_ratio = _body(candle) / total

    if body_ratio <= 0.10:
        return CandlePattern("Doji", "neutral", round(1 - body_ratio, 2),
                              "Indecision candle — watch for follow-through")
    return None


def _check_rejection_candle(candle: pd.Series) -> Optional[CandlePattern]:
    """
    Strong rejection: large body (> 60% range) with a wick on one side showing failure.
    Bullish rejection: closed near high, tiny upper wick — strong close.
    Bearish rejection: closed near low, tiny lower wick — strong close.
    """
    total = _total_range(candle)
    if total == 0:
        return None

    body  = _body(candle)
    upper = _upper_wick(candle)
    lower = _lower_wick(candle)

    body_ratio  = body  / total
    upper_ratio = upper / total
    lower_ratio = lower / total

    if body_ratio < 0.40:
        return None

    if _is_bullish(candle) and lower_ratio >= 0.15 and upper_ratio <= 0.20:
        return CandlePattern("Bullish Rejection", "bullish", round(body_ratio, 2),
                              "Strong bullish close with lower wick rejection")

    if _is_bearish(candle) and upper_ratio >= 0.15 and lower_ratio <= 0.20:
        return CandlePattern("Bearish Rejection", "bearish", round(body_ratio, 2),
                              "Strong bearish close with upper wick rejection")

    return None


def _check_marubozu(candle: pd.Series) -> Optional[CandlePattern]:
    """
    Marubozu: full-body candle, almost no wicks (< 5% each side).
    Indicates very strong momentum.
    """
    total = _total_range(candle)
    if total == 0:
        return None

    upper = _upper_wick(candle) / total
    lower = _lower_wick(candle) / total

    if upper <= 0.05 and lower <= 0.05:
        if _is_bullish(candle):
            return CandlePattern("Bullish Marubozu", "bullish", 0.90,
                                  "Strong bullish momentum candle with no wicks")
        else:
            return CandlePattern("Bearish Marubozu", "bearish", 0.90,
                                  "Strong bearish momentum candle with no wicks")
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyse_price_action(df: pd.DataFrame) -> PriceActionResult:
    """
    Analyse the last few candles and return detected patterns.
    Primary focus: most recent completed candle and its predecessor.
    """
    result = PriceActionResult()

    if len(df) < 3:
        return result

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    patterns: List[CandlePattern] = []

    # 2-candle patterns first
    eng = _check_engulfing(curr, prev)
    if eng:
        patterns.append(eng)

    # 1-candle patterns on latest candle
    for checker in [_check_pin_bar, _check_doji, _check_rejection_candle, _check_marubozu]:
        p = checker(curr)
        if p:
            patterns.append(p)

    # Also check previous candle for pin bars / rejection (still in play)
    for checker in [_check_pin_bar, _check_rejection_candle]:
        p = checker(prev)
        if p:
            # Downgrade strength slightly — it's the prior candle
            p.strength = round(p.strength * 0.8, 2)
            p.name     = p.name + " (prior)"
            patterns.append(p)

    result.patterns = patterns

    # Composite bias
    bull = sum(p.strength for p in patterns if p.direction == "bullish")
    bear = sum(p.strength for p in patterns if p.direction == "bearish")

    # Normalise (cap at 1.0)
    result.bullish_bias = round(min(bull, 1.0), 3)
    result.bearish_bias = round(min(bear, 1.0), 3)

    # Dominant pattern = highest strength non-neutral
    non_neutral = [p for p in patterns if p.direction != "neutral"]
    if non_neutral:
        result.dominant_pattern = max(non_neutral, key=lambda p: p.strength)

    result.pattern_names = [p.name for p in patterns]

    return result

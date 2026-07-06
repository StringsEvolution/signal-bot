"""
Indicator Module — Confirmation only.
EMA 50/200, RSI(14), MACD, ATR(14).
Indicators do NOT generate signals alone — they confirm structure signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class IndicatorResult:
    # EMA
    ema50:  float = 0.0
    ema200: float = 0.0
    ema_trend: str = "neutral"     # "bullish" | "bearish" | "neutral"
    ema_spread_pct: float = 0.0    # % gap between EMA50 and EMA200

    # RSI
    rsi: float = 50.0
    rsi_zone: str = "neutral"      # "overbought" | "oversold" | "neutral"
    rsi_recovering: bool = False   # RSI was oversold/overbought and now recovering
    rsi_recovery_dir: str = "none" # "bullish" (up from oversold) | "bearish" (down from overbought) | "none"

    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_cross: Optional[str] = None  # "bullish" | "bearish" | None
    macd_above_zero: bool = False

    # ATR
    atr: float = 0.0
    atr_pct: float = 0.0           # ATR as % of price
    volatility_state: str = "normal"  # "low" | "normal" | "high"

    # Composite alignment score (–1 to +1)
    bull_alignment: float = 0.0
    bear_alignment: float = 0.0


# ---------------------------------------------------------------------------
# Individual indicator calculations
# ---------------------------------------------------------------------------

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    k = 2 / (period + 1)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    out    = np.full(len(closes), np.nan)
    avg_g  = np.mean(gains[:period])
    avg_l  = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs    = avg_g / avg_l if avg_l != 0 else 1e9
        out[i + 1] = 100 - (100 / (1 + rs))

    return out


def _macd(closes: np.ndarray,
          fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(np.where(np.isnan(macd_line), 0, macd_line), signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )

    atr_vals = np.zeros(len(df))
    if len(tr) < period:
        return atr_vals

    atr_vals[period] = np.mean(tr[1: period + 1])
    for i in range(period + 1, len(tr)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

    return atr_vals


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> IndicatorResult:
    """
    Compute all indicators and return a structured result.
    Requires at least 220 candles for EMA200.
    """
    closes = df["close"].values.astype(float)
    result = IndicatorResult()

    if len(df) < 30:
        return result

    # ----- EMA -----
    ema50_arr  = _ema(closes, 50)
    ema200_arr = _ema(closes, 200)

    result.ema50  = round(float(ema50_arr[-1]),  6)
    result.ema200 = round(float(ema200_arr[-1]), 6)

    if not (np.isnan(result.ema50) or np.isnan(result.ema200)):
        spread = abs(result.ema50 - result.ema200) / result.ema200
        result.ema_spread_pct = round(spread * 100, 4)

        if result.ema50 > result.ema200 and spread > 0.001:
            result.ema_trend = "bullish"
        elif result.ema50 < result.ema200 and spread > 0.001:
            result.ema_trend = "bearish"
        else:
            result.ema_trend = "neutral"
    else:
        result.ema_trend = "neutral"

    # ----- RSI -----
    rsi_arr   = _rsi(closes, 14)
    result.rsi = round(float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0, 2)

    if result.rsi >= 70:
        result.rsi_zone = "overbought"
    elif result.rsi <= 30:
        result.rsi_zone = "oversold"
    else:
        result.rsi_zone = "neutral"

    # Recovering = was extreme, now moving back toward 50.
    # Record the DIRECTION of recovery so downstream alignment gets the sign
    # right: coming up from oversold is bullish; coming down from overbought
    # is bearish. (Previously the alignment block inferred direction from the
    # CURRENT zone, which mis-signed a recovery that had already climbed past
    # 45 — scoring an oversold->up bounce as bearish.)
    if len(rsi_arr) >= 3:
        prev_rsi = float(rsi_arr[-3]) if not np.isnan(rsi_arr[-3]) else 50.0
        if prev_rsi <= 30 and result.rsi > prev_rsi:
            result.rsi_recovering   = True
            result.rsi_recovery_dir = "bullish"     # was oversold, rising
        elif prev_rsi >= 70 and result.rsi < prev_rsi:
            result.rsi_recovering   = True
            result.rsi_recovery_dir = "bearish"     # was overbought, falling
        else:
            result.rsi_recovering   = False
            result.rsi_recovery_dir = "none"

    # ----- MACD -----
    macd_line, signal_line, histogram = _macd(closes)
    result.macd_line   = round(float(macd_line[-1])   if not np.isnan(macd_line[-1])   else 0.0, 6)
    result.macd_signal = round(float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0.0, 6)
    result.macd_hist   = round(float(histogram[-1])   if not np.isnan(histogram[-1])   else 0.0, 6)
    result.macd_above_zero = result.macd_line > 0

    # Crossover detection (last 2 bars)
    if len(macd_line) >= 2:
        m_prev = float(macd_line[-2])   if not np.isnan(macd_line[-2])   else 0.0
        s_prev = float(signal_line[-2]) if not np.isnan(signal_line[-2]) else 0.0
        m_curr = result.macd_line
        s_curr = result.macd_signal

        if m_prev < s_prev and m_curr > s_curr:
            result.macd_cross = "bullish"
        elif m_prev > s_prev and m_curr < s_curr:
            result.macd_cross = "bearish"

    # ----- ATR -----
    atr_arr = _atr(df, 14)
    result.atr = round(float(atr_arr[-1]) if atr_arr[-1] > 0 else 0.0, 6)
    result.atr_pct = round(result.atr / closes[-1] * 100, 4) if closes[-1] > 0 else 0.0

    # Volatility classification (asset-agnostic using ATR%)
    if result.atr_pct < 0.05:
        result.volatility_state = "low"
    elif result.atr_pct > 0.25:
        result.volatility_state = "high"
    else:
        result.volatility_state = "normal"

    # ----- Composite alignment -----
    bull = 0.0
    bear = 0.0

    # EMA
    if result.ema_trend == "bullish":
        bull += 0.3
    elif result.ema_trend == "bearish":
        bear += 0.3

    # RSI
    if result.rsi_zone == "oversold":
        bull += 0.2
    elif result.rsi_zone == "overbought":
        bear += 0.2
    elif 45 <= result.rsi <= 55:
        pass  # neutral
    elif result.rsi > 55:
        bull += 0.1
    else:
        bear += 0.1

    # RSI recovering — sign follows the direction of recovery, not the
    # current zone (a bounce up from oversold stays bullish even after RSI
    # has climbed back above 45).
    if result.rsi_recovering:
        if result.rsi_recovery_dir == "bullish":
            bull += 0.15
        elif result.rsi_recovery_dir == "bearish":
            bear += 0.15

    # MACD
    if result.macd_cross == "bullish":
        bull += 0.25
    elif result.macd_cross == "bearish":
        bear += 0.25
    elif result.macd_hist > 0 and result.macd_above_zero:
        bull += 0.1
    elif result.macd_hist < 0 and not result.macd_above_zero:
        bear += 0.1

    result.bull_alignment = round(min(bull, 1.0), 3)
    result.bear_alignment = round(min(bear, 1.0), 3)

    return result

"""
AI Confidence Engine
Uses LightGBM (with XGBoost fallback) to score signal quality.
Outputs probability UP, probability DOWN, and a 0–100 confidence score.
Confidence threshold is now timeframe-aware.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "signal_model.pkl")

# Global default threshold (used by backtester and external references)
CONFIDENCE_THRESHOLD = 80.0

# Timeframe-aware thresholds — shorter TFs get lower bar since AI trained on M5/M15
TF_CONFIDENCE_THRESHOLD = {
    "M1":  65.0,
    "M2":  60.0,
    "M3":  60.0,
    "M5":  80.0,
    "M15": 80.0,
}

def get_confidence_threshold(timeframe: str) -> float:
    return TF_CONFIDENCE_THRESHOLD.get(timeframe, CONFIDENCE_THRESHOLD)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "rsi", "rsi_zone_enc",
    "macd_hist", "macd_cross_enc",
    "ema_trend_enc", "ema_spread_pct",
    "atr_pct", "volatility_enc",
    "sr_distance_pct",
    "at_support", "at_resistance",
    "pullback", "breakout_enc",
    "bull_pa", "bear_pa",
    "trend_strength", "trend_enc",
    "session_enc",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
]


def _encode_session(dt: datetime) -> int:
    h = dt.hour
    if 7 <= h < 16:
        return 1
    if 13 <= h < 22:
        return 2
    if 7 <= h < 13:
        return 3
    return 0


def _cyclic(val: float, max_val: float) -> Tuple[float, float]:
    rad = 2 * np.pi * val / max_val
    return np.sin(rad), np.cos(rad)


def build_features(
    rsi: float,
    macd_hist: float,
    macd_cross: Optional[str],
    ema_trend: str,
    ema_spread_pct: float,
    atr_pct: float,
    volatility_state: str,
    sr_distance_pct: float,
    at_support: bool,
    at_resistance: bool,
    pullback: bool,
    breakout: Optional[str],
    bull_pa: float,
    bear_pa: float,
    trend_strength: float,
    trend: str,
    dt: Optional[datetime] = None,
) -> np.ndarray:
    if dt is None:
        dt = datetime.utcnow()

    rsi_zone_enc   = 0
    if rsi <= 30:  rsi_zone_enc = -1
    elif rsi >= 70: rsi_zone_enc = 1

    macd_cross_enc = {"bullish": 1, "bearish": -1, None: 0}.get(macd_cross, 0)
    ema_trend_enc  = {"bullish": 1, "bearish": -1, "neutral": 0}.get(ema_trend, 0)
    volatility_enc = {"low": -1, "normal": 0, "high": 1}.get(volatility_state, 0)
    breakout_enc   = {"bullish_break": 1, "bearish_break": -1, None: 0}.get(breakout, 0)
    trend_enc      = {"bullish": 1, "bearish": -1, "ranging": 0}.get(trend, 0)
    session_enc    = _encode_session(dt)

    h_sin, h_cos = _cyclic(dt.hour, 24)
    d_sin, d_cos = _cyclic(dt.weekday(), 7)

    feats = np.array([
        rsi, rsi_zone_enc,
        macd_hist, macd_cross_enc,
        ema_trend_enc, ema_spread_pct,
        atr_pct, volatility_enc,
        sr_distance_pct,
        int(at_support), int(at_resistance),
        int(pullback), breakout_enc,
        bull_pa, bear_pa,
        trend_strength, trend_enc,
        session_enc,
        h_sin, h_cos,
        d_sin, d_cos,
    ], dtype=float)

    return feats


# ---------------------------------------------------------------------------
# Model loader / trainer
# ---------------------------------------------------------------------------

class AIConfidenceEngine:
    def __init__(self):
        self.model     = None
        self.is_fitted = False
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self.is_fitted = True
                logger.info("Loaded pre-trained model from disk.")
                return
            except Exception as exc:
                logger.warning(f"Could not load model ({exc}), will use heuristic mode.")
        self._init_model()

    def _init_model(self):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            logger.info("XGBoost model initialised.")
        except (ImportError, Exception) as e:
            logger.warning(f"XGBoost not available ({e}), trying sklearn...")
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
                )
                logger.info("GradientBoosting (sklearn) model initialised.")
            except Exception as e2:
                logger.warning(f"sklearn GradientBoosting failed ({e2}), using heuristic mode only.")
                self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True

        preds = self.model.predict(X_val)
        logger.info("Validation report:\n" + classification_report(y_val, preds))

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {MODEL_PATH}")

    def score(self, features: np.ndarray) -> "AIScore":
        if self.is_fitted and self.model is not None:
            try:
                X    = features.reshape(1, -1)
                proba = self.model.predict_proba(X)[0]
                if len(proba) == 2:
                    prob_up   = float(proba[1])
                    prob_down = float(proba[0])
                else:
                    prob_up   = float(proba[0])
                    prob_down = 1 - prob_up

                confidence = round((max(prob_up, prob_down) - 0.5) * 200, 1)
                confidence = min(max(confidence, 0.0), 100.0)

                return AIScore(
                    prob_up=round(prob_up, 4),
                    prob_down=round(prob_down, 4),
                    confidence=confidence,
                    direction="CALL" if prob_up > prob_down else "PUT",
                    model_mode="ml",
                )
            except Exception as exc:
                logger.warning(f"ML scoring failed: {exc}, falling back to heuristic.")

        return self._heuristic_score(features)

    def _heuristic_score(self, features: np.ndarray) -> "AIScore":
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        f   = features

        rsi        = f[idx["rsi"]]
        rsi_zone   = f[idx["rsi_zone_enc"]]
        macd_cross = f[idx["macd_cross_enc"]]
        ema_trend  = f[idx["ema_trend_enc"]]
        sr_dist    = f[idx["sr_distance_pct"]]
        at_sup     = f[idx["at_support"]]
        at_res     = f[idx["at_resistance"]]
        pullback   = f[idx["pullback"]]
        bull_pa    = f[idx["bull_pa"]]
        bear_pa    = f[idx["bear_pa"]]
        ts         = f[idx["trend_strength"]]
        trend_enc  = f[idx["trend_enc"]]
        volatility = f[idx["volatility_enc"]]

        bull = 0.0
        bear = 0.0

        if ema_trend == 1:   bull += 0.25
        if ema_trend == -1:  bear += 0.25

        if rsi_zone == -1:   bull += 0.20
        if rsi_zone ==  1:   bear += 0.20

        if macd_cross ==  1: bull += 0.25
        if macd_cross == -1: bear += 0.25

        if at_sup == 1 and trend_enc == 1:   bull += 0.20
        if at_res == 1 and trend_enc == -1:  bear += 0.20
        if at_sup == 1 and trend_enc == -1:  bear += 0.10
        if at_res == 1 and trend_enc == 1:   bull += 0.10

        if pullback == 1 and trend_enc == 1:  bull += 0.15
        if pullback == 1 and trend_enc == -1: bear += 0.15

        bull += bull_pa * 0.30
        bear += bear_pa * 0.30

        if trend_enc == 1:  bull += ts * 0.15
        if trend_enc == -1: bear += ts * 0.15

        if volatility == -1:
            bull *= 0.5
            bear *= 0.5
        if sr_dist > 0.5:
            bull *= 0.7
            bear *= 0.7

        bull += 0.05
        bear += 0.05

        total = bull + bear
        if total == 0:
            return AIScore(0.5, 0.5, 0.0, "PUT", "heuristic")

        prob_up    = bull / total
        prob_down  = bear / total
        confidence = round((max(prob_up, prob_down) - 0.5) * 200, 1)
        confidence = min(max(confidence, 0.0), 100.0)

        return AIScore(
            prob_up=round(prob_up, 4),
            prob_down=round(prob_down, 4),
            confidence=confidence,
            direction="CALL" if prob_up > prob_down else "PUT",
            model_mode="heuristic",
        )


@dataclass
class AIScore:
    prob_up:    float
    prob_down:  float
    confidence: float
    direction:  str
    model_mode: str

    @property
    def passes_threshold(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(signals_log: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    required = set(FEATURE_NAMES + ["result"])
    missing  = required - set(signals_log.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")
    X = signals_log[FEATURE_NAMES].values.astype(float)
    y = (signals_log["result"] == "win").astype(int).values
    return X, y


_engine: Optional[AIConfidenceEngine] = None

def get_ai_engine() -> AIConfidenceEngine:
    global _engine
    if _engine is None:
        _engine = AIConfidenceEngine()
    return _engine

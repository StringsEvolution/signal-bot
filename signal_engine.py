"""
Signal Engine — Orchestrates all modules.
A signal is only emitted when ALL conditions are satisfied:
  1. Filters pass (session, news, volatility)
  2. Market structure confirmed
  3. Price action pattern detected
  4. Direction resolved (votes decisive)
  5. AI confidence > timeframe threshold
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd

from data_engine import fetch_ohlc, store_ohlc, load_ohlc, TF_EXPIRY, ASSETS, TIMEFRAMES
from market_structure import analyse_structure, StructureResult
from indicators import compute_indicators, IndicatorResult
from price_action import analyse_price_action, PriceActionResult
from ai_model import build_features, get_ai_engine, AIScore, CONFIDENCE_THRESHOLD, get_confidence_threshold
from filter_engine import apply_filters, get_current_session

logger = logging.getLogger(__name__)

MAX_DAILY_SIGNALS    = 20
SIGNAL_COOLDOWN_MINS = 5

# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    timestamp:   datetime
    asset:       str
    timeframe:   str
    direction:   str
    entry_price: float
    expiry_min:  int
    confidence:  float
    reasons:     List[str] = field(default_factory=list)
    warnings:    List[str] = field(default_factory=list)
    session:     str       = ""
    prob_up:     float     = 0.0
    prob_down:   float     = 0.0
    model_mode:  str       = ""
    is_valid:    bool      = True

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp.isoformat(),
            "asset":       self.asset,
            "timeframe":   self.timeframe,
            "direction":   self.direction,
            "entry_price": self.entry_price,
            "expiry_min":  self.expiry_min,
            "confidence":  self.confidence,
            "reasons":     self.reasons,
            "warnings":    self.warnings,
            "session":     self.session,
            "prob_up":     self.prob_up,
            "prob_down":   self.prob_down,
        }

    def format_message(self, vip: bool = False) -> str:
        star = "⭐ VIP" if vip else "🔔 Signal"
        bar  = "🟢" if self.direction == "CALL" else "🔴"
        reasons_text = "\n".join(f"  ✅ {r}" for r in self.reasons)
        warn_text    = ("\n" + "\n".join(f"  ⚠️ {w}" for w in self.warnings)) if self.warnings else ""
        msg = f"""
{star} | {bar} {self.direction}

📊 Pair:       {self.asset}
🕐 Timeframe:  {self.timeframe}
⏳ Expiry:     {self.expiry_min} minutes
💰 Entry:      {self.entry_price:.5f}
🤖 Confidence: {self.confidence:.0f}%
🌍 Session:    {self.session}
🕒 Time (UTC): {self.timestamp.strftime('%H:%M:%S')}

📋 Reasons:
{reasons_text}{warn_text}

⚠️ Risk Disclaimer: Binary options carry significant risk. Trade only what you can afford to lose.
"""
        if not vip:
            msg += "\n🔒 Full analysis in VIP channel. Join: t.me/your_vip_channel"
        return msg.strip()


# ---------------------------------------------------------------------------
# Daily counter + deduplication
# ---------------------------------------------------------------------------

_daily_counts: dict = {}
_last_emitted: dict = {}

def _daily_count_key(asset: str, dt: datetime) -> str:
    return f"{asset}:{dt.strftime('%Y-%m-%d')}"

def _increment_count(asset: str, dt: datetime) -> int:
    key = _daily_count_key(asset, dt)
    _daily_counts[key] = _daily_counts.get(key, 0) + 1
    return _daily_counts[key]

def _get_count(asset: str, dt: datetime) -> int:
    key = _daily_count_key(asset, dt)
    return _daily_counts.get(key, 0)

def _is_duplicate(asset: str, timeframe: str, direction: str, dt: datetime) -> bool:
    key  = f"{asset}/{timeframe}/{direction}"
    last = _last_emitted.get(key)
    if last and (dt - last) < timedelta(minutes=SIGNAL_COOLDOWN_MINS):
        logger.info(
            f"⛔ {asset}/{timeframe}: DUPLICATE blocked — "
            f"{direction} already sent {(dt - last).seconds // 60}m ago "
            f"(cooldown={SIGNAL_COOLDOWN_MINS}m)"
        )
        return True
    return False

def _mark_emitted(asset: str, timeframe: str, direction: str, dt: datetime):
    _last_emitted[f"{asset}/{timeframe}/{direction}"] = dt


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def generate_signal(
    asset: str,
    timeframe: str,
    df: Optional[pd.DataFrame] = None,
    dt: Optional[datetime] = None,
    spread_pct: float = 0.0,
) -> Optional[Signal]:

    dt  = dt or datetime.utcnow()
    tag = f"{asset}/{timeframe}"

    # Timeframe-aware confidence threshold
    tf_threshold = get_confidence_threshold(timeframe)

    # --- Daily cap
    if _get_count(asset, dt) >= MAX_DAILY_SIGNALS:
        logger.info(f"⛔ {tag}: daily cap reached ({MAX_DAILY_SIGNALS})")
        return None

    # --- Load data
    if df is None or len(df) < 50:
        try:
            df = load_ohlc(asset, timeframe, limit=300)
            if len(df) < 50:
                df = fetch_ohlc(asset, timeframe, n_candles=300)
                store_ohlc(asset, timeframe, df)
        except Exception as exc:
            logger.warning(f"⛔ {tag}: data fetch failed — {exc}")
            return None

    if len(df) < 50:
        logger.info(f"⛔ {tag}: insufficient data ({len(df)} candles)")
        return None

    entry_price = float(df["close"].iloc[-1])

    # ---- Gate 1: Indicators ----
    ind = compute_indicators(df)

    # ---- Gate 2: Filters ----
    filt = apply_filters(
        asset=asset,
        timeframe=timeframe,
        atr_pct=ind.atr_pct,
        volatility_state=ind.volatility_state,
        trend=ind.ema_trend,
        dt=dt,
        spread_pct=spread_pct,
    )
    if not filt.allowed:
        logger.info(f"⛔ {tag}: FILTER blocked — {filt.reasons[0] if filt.reasons else '?'}")
        return None

    # ---- Gate 3: Market structure ----
    struct = analyse_structure(df)

    logger.info(
        f"🔍 {tag}: trend={struct.trend}({struct.trend_strength:.0%}) "
        f"valid={struct.structure_valid} | "
        f"at_sup={struct.at_support} at_res={struct.at_resistance} "
        f"pullback={struct.pullback} fakeout={struct.fakeout}"
    )

    if not struct.structure_valid:
        logger.info(
            f"⛔ {tag}: STRUCTURE blocked — "
            f"trend={struct.trend} strength={struct.trend_strength:.0%} "
            f"at_level={struct.at_support or struct.at_resistance or struct.pullback} "
            f"fakeout={struct.fakeout}"
        )
        return None

    # ---- Gate 4: Indicators detail ----
    logger.info(
        f"📊 {tag}: EMA={ind.ema_trend}(spread={ind.ema_spread_pct:.3f}%) "
        f"RSI={ind.rsi:.1f}[{ind.rsi_zone}] "
        f"MACD_cross={ind.macd_cross} MACD_hist={ind.macd_hist:.6f} "
        f"ATR%={ind.atr_pct:.4f} vol={ind.volatility_state}"
    )

    # ---- Gate 5: Price action ----
    pa = analyse_price_action(df)
    logger.info(
        f"🕯️  {tag}: PA patterns={pa.pattern_names} "
        f"bull={pa.bullish_bias:.2f} bear={pa.bearish_bias:.2f}"
    )

    if pa.dominant_pattern is None:
        logger.info(f"⛔ {tag}: PA blocked — no directional candle pattern detected")
        return None

    # ---- Gate 6: Direction vote ----
    direction, votes = _resolve_direction_verbose(struct, ind, pa)
    logger.info(
        f"🗳️  {tag}: votes CALL={votes['CALL']} PUT={votes['PUT']} → direction={direction}"
    )

    if direction is None:
        total = votes["CALL"] + votes["PUT"]
        ratio = max(votes["CALL"], votes["PUT"]) / total if total > 0 else 0
        logger.info(
            f"⛔ {tag}: DIRECTION blocked — no decisive majority "
            f"(best={ratio:.0%}, need 65%)"
        )
        return None

    # ---- Gate 7: Duplicate check ----
    if _is_duplicate(asset, timeframe, direction, dt):
        return None

    # ---- Gate 8: AI confidence ----
    features = build_features(
        rsi=ind.rsi,
        macd_hist=ind.macd_hist,
        macd_cross=ind.macd_cross,
        ema_trend=ind.ema_trend,
        ema_spread_pct=ind.ema_spread_pct,
        atr_pct=ind.atr_pct,
        volatility_state=ind.volatility_state,
        sr_distance_pct=struct.sr_distance_pct,
        at_support=struct.at_support,
        at_resistance=struct.at_resistance,
        pullback=struct.pullback,
        breakout=struct.breakout,
        bull_pa=pa.bullish_bias,
        bear_pa=pa.bearish_bias,
        trend_strength=struct.trend_strength,
        trend=struct.trend,
        dt=dt,
    )

    ai_score = get_ai_engine().score(features)
    logger.info(
        f"🤖 {tag}: AI dir={ai_score.direction} conf={ai_score.confidence:.1f}% "
        f"P(UP)={ai_score.prob_up:.2f} P(DN)={ai_score.prob_down:.2f} "
        f"threshold={tf_threshold:.0f}% passes={ai_score.confidence >= tf_threshold}"
    )

    # Direction consistency check
    if ai_score.direction != direction:
        matching_prob  = ai_score.prob_up if direction == "CALL" else ai_score.prob_down
        min_prob       = 0.55 if timeframe in ("M1", "M2", "M3") else 0.60
        logger.info(
            f"⚠️  {tag}: AI direction conflict "
            f"(AI={ai_score.direction} struct={direction}) "
            f"matching_prob={matching_prob:.2f}"
        )
        if matching_prob < min_prob:
            logger.info(f"⛔ {tag}: AI CONFLICT blocked — matching_prob {matching_prob:.2f} < {min_prob}")
            return None

    if ai_score.confidence < tf_threshold:
        logger.info(
            f"⛔ {tag}: AI CONFIDENCE blocked — "
            f"{ai_score.confidence:.1f}% < {tf_threshold:.0f}% threshold for {timeframe}"
        )
        return None

    # ---- All gates passed — emit signal ----
    reasons = _build_reasons(direction, struct, ind, pa, ai_score, dt)
    expiry  = TF_EXPIRY[timeframe]
    session = get_current_session(dt)

    signal = Signal(
        timestamp=dt,
        asset=asset,
        timeframe=timeframe,
        direction=direction,
        entry_price=round(entry_price, 6),
        expiry_min=expiry,
        confidence=ai_score.confidence,
        reasons=reasons,
        warnings=filt.warnings,
        session=session,
        prob_up=ai_score.prob_up,
        prob_down=ai_score.prob_down,
        model_mode=ai_score.model_mode,
    )

    _increment_count(asset, dt)
    _mark_emitted(asset, timeframe, direction, dt)

    logger.info(
        f"✅ SIGNAL EMITTED: {asset}/{timeframe} {direction} "
        f"conf={ai_score.confidence:.1f}% entry={entry_price:.5f}"
    )
    return signal


# ---------------------------------------------------------------------------
# Scan all
# ---------------------------------------------------------------------------

def scan_all(dt: Optional[datetime] = None, spread_pcts: dict = {}) -> List[Signal]:
    signals = []
    dt      = dt or datetime.utcnow()

    logger.info(f"━━━ Scan started {dt.strftime('%H:%M:%S UTC')} ━━━")

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            spread = spread_pcts.get(asset, 0.0)
            try:
                sig = generate_signal(asset, tf, dt=dt, spread_pct=spread)
                if sig:
                    signals.append(sig)
            except Exception as exc:
                logger.error(f"Scan error {asset}/{tf}: {exc}", exc_info=True)

    logger.info(f"━━━ Scan complete — {len(signals)} signal(s) ━━━")
    return signals


# ---------------------------------------------------------------------------
# Direction resolver
# ---------------------------------------------------------------------------

def _resolve_direction_verbose(
    struct: StructureResult,
    ind: IndicatorResult,
    pa: PriceActionResult,
) -> Tuple[Optional[str], dict]:
    votes = {"CALL": 0, "PUT": 0}

    if struct.trend == "bullish":
        if struct.at_support:  votes["CALL"] += 3
        if struct.pullback:    votes["CALL"] += 2
    elif struct.trend == "bearish":
        if struct.at_resistance: votes["PUT"] += 3
        if struct.pullback:      votes["PUT"] += 2

    if struct.breakout == "bullish_break": votes["CALL"] += 2
    elif struct.breakout == "bearish_break": votes["PUT"] += 2

    if ind.ema_trend == "bullish":  votes["CALL"] += 1
    elif ind.ema_trend == "bearish": votes["PUT"]  += 1

    if ind.rsi_zone == "oversold":    votes["CALL"] += 2
    elif ind.rsi_zone == "overbought": votes["PUT"]  += 2
    elif ind.rsi > 55: votes["CALL"] += 1
    elif ind.rsi < 45: votes["PUT"]  += 1

    if ind.macd_cross == "bullish":  votes["CALL"] += 2
    elif ind.macd_cross == "bearish": votes["PUT"]  += 2
    elif ind.macd_hist > 0: votes["CALL"] += 1
    elif ind.macd_hist < 0: votes["PUT"]  += 1

    if pa.dominant_pattern:
        if pa.dominant_pattern.direction == "bullish":
            votes["CALL"] += int(pa.dominant_pattern.strength * 3)
        elif pa.dominant_pattern.direction == "bearish":
            votes["PUT"]  += int(pa.dominant_pattern.strength * 3)

    total = votes["CALL"] + votes["PUT"]
    if total == 0:
        return None, votes

    if votes["CALL"] / total >= 0.65:
        return "CALL", votes
    elif votes["PUT"] / total >= 0.65:
        return "PUT", votes

    return None, votes


def _resolve_direction(struct, ind, pa):
    direction, _ = _resolve_direction_verbose(struct, ind, pa)
    return direction


def _build_reasons(direction, struct, ind, pa, ai, dt) -> List[str]:
    reasons = []

    reasons.append(
        f"{'Uptrend' if struct.trend == 'bullish' else 'Downtrend'} confirmed "
        f"(EMA 50 {'>' if struct.trend == 'bullish' else '<'} EMA 200, "
        f"strength={struct.trend_strength:.0%})"
    )

    if struct.at_support and direction == "CALL":
        reasons.append(f"Price at key support zone ({struct.nearest_support:.5f})")
    if struct.at_resistance and direction == "PUT":
        reasons.append(f"Price at key resistance zone ({struct.nearest_resistance:.5f})")
    if struct.pullback:
        reasons.append(f"Clean {'bullish' if direction == 'CALL' else 'bearish'} pullback to key level")
    if struct.breakout:
        reasons.append(f"Structural breakout: {struct.breakout.replace('_', ' ').title()}")

    if pa.dominant_pattern:
        reasons.append(f"{pa.dominant_pattern.name} — {pa.dominant_pattern.description}")

    if ind.rsi_zone == "oversold" and direction == "CALL":
        reasons.append(f"RSI({ind.rsi:.1f}) recovering from oversold zone")
    elif ind.rsi_zone == "overbought" and direction == "PUT":
        reasons.append(f"RSI({ind.rsi:.1f}) retreating from overbought zone")
    else:
        reasons.append(f"RSI at {ind.rsi:.1f} — aligned with {direction.lower()} bias")

    if ind.macd_cross:
        reasons.append(f"MACD {ind.macd_cross} crossover confirmed")
    elif ind.macd_hist > 0 and direction == "CALL":
        reasons.append("MACD histogram positive — bullish momentum building")
    elif ind.macd_hist < 0 and direction == "PUT":
        reasons.append("MACD histogram negative — bearish momentum building")

    reasons.append(
        f"AI confidence: {ai.confidence:.0f}% "
        f"({'ML' if ai.model_mode == 'ml' else 'Heuristic'} | "
        f"P(UP)={ai.prob_up:.1%} P(DN)={ai.prob_down:.1%})"
    )

    reasons.append(f"Session: {get_current_session(dt)}")

    return reasons

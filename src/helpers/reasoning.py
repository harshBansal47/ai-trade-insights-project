import numpy as np
import pandas as pd
from typing import Optional
from src.constants import TF_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers / Guards
# ---------------------------------------------------------------------------
def _validate_df(df: pd.DataFrame, required_cols: list[str], fn_name: str) -> None:
    """Raise early with a clear message if the DataFrame is unusable."""
    if df is None or df.empty:
        raise ValueError(f"[{fn_name}] DataFrame is None or empty.")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{fn_name}] Missing columns: {missing}")


# ---------------------------------------------------------------------------
# Fake Signal Detection
# ---------------------------------------------------------------------------
def detect_fake_signal(tf_features: dict) -> dict:
    """
    Detect whether a signal is likely fake based on indicator conflicts.

    Parameters
    ----------
    tf_features : dict
        Output of build_timeframe_features(). Must include 'bb' key.

    Returns
    -------
    dict with keys:
        is_fake        : bool  — True if >= 2 directional conflicts exist,
                                 or >= 1 conflict + low_volatility
        conflicts      : list  — specific conflict labels
        low_volatility : bool  — ATR contracting (not a directional conflict)
    """
    trend        = tf_features["trend"]["direction"]
    rsi_zone     = tf_features["rsi"]["zone"]
    macd_signal  = tf_features["macd"]["signal"]
    atr_state    = tf_features["atr"]["state"]
    bb_position  = tf_features["bb"]["position"]
    bb_squeeze   = tf_features["bb"]["squeeze"]

    conflicts: list[str] = []

    # RSI vs Trend
    if trend == "UPTREND" and rsi_zone == "overbought":
        conflicts.append("rsi_overbought_in_uptrend")

    if trend == "DOWNTREND" and rsi_zone == "oversold":
        conflicts.append("rsi_oversold_in_downtrend")

    # MACD vs Trend
    if trend == "UPTREND" and "bearish" in macd_signal:
        conflicts.append("macd_bearish_against_trend")

    if trend == "DOWNTREND" and "bullish" in macd_signal:
        conflicts.append("macd_bullish_against_trend")

    # BB vs Trend — price outside band against trend direction is suspect
    # e.g. price above upper band in a DOWNTREND = overextended fake bounce
    if trend == "DOWNTREND" and bb_position == "above_upper":
        conflicts.append("bb_above_upper_in_downtrend")

    if trend == "UPTREND" and bb_position == "below_lower":
        conflicts.append("bb_below_lower_in_uptrend")

    # BB squeeze — a signal fired during a squeeze is unreliable because
    # bands are compressed and direction is not yet confirmed
    if bb_squeeze:
        conflicts.append("bb_squeeze_signal")

    # ATR contraction is a separate risk flag — not a directional conflict
    is_low_vol = atr_state == "contracting"

    # Fake if >= 2 directional conflicts, or >= 1 conflict + low volatility
    is_fake = len(conflicts) >= 2 or (len(conflicts) >= 1 and is_low_vol)

    return {
        "is_fake":       is_fake,
        "conflicts":     conflicts,
        "low_volatility": is_low_vol,
    }


# ---------------------------------------------------------------------------
# Swing Points
# ---------------------------------------------------------------------------
def find_swing_points(
    df: pd.DataFrame,
    window: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Identify swing highs and lows using a rolling window comparison.

    Note: The first and last `window` rows are excluded by design — there
    are not enough neighbours to confirm a swing at the boundaries.
    """
    _validate_df(df, ["high", "low"], "find_swing_points")

    highs: list[float] = []
    lows:  list[float] = []

    for i in range(window, len(df) - window):
        window_slice_high = df["high"].iloc[i - window: i + window + 1]
        window_slice_low  = df["low"].iloc[i - window: i + window + 1]

        if df["high"].iloc[i] == window_slice_high.max():
            highs.append(float(df["high"].iloc[i]))

        if df["low"].iloc[i] == window_slice_low.min():
            lows.append(float(df["low"].iloc[i]))

    return highs, lows


# ---------------------------------------------------------------------------
# Level Clustering
# ---------------------------------------------------------------------------
def cluster_levels(levels: list[float], tolerance: float = 0.005) -> list[dict]:
    """
    Group nearby price levels into clusters.

    Levels are sorted first to ensure order-independent assignment.
    """
    if not levels:
        return []

    clusters: list[dict] = []

    for level in sorted(levels):
        placed = False
        for cluster in clusters:
            if abs(level - cluster["mean"]) / cluster["mean"] < tolerance:
                cluster["values"].append(level)
                cluster["mean"] = float(np.mean(cluster["values"]))
                placed = True
                break

        if not placed:
            clusters.append({"mean": level, "values": [level]})

    return clusters


# ---------------------------------------------------------------------------
# Support & Resistance Zones
# ---------------------------------------------------------------------------
def get_sr_zones(df: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Compute the top N support and resistance zones by cluster strength.
    """
    _validate_df(df, ["high", "low"], "get_sr_zones")

    highs, lows = find_swing_points(df)

    resistance_clusters = cluster_levels(highs)
    support_clusters    = cluster_levels(lows)

    resistance = sorted(
        [{"level": c["mean"], "strength": len(c["values"])} for c in resistance_clusters],
        key=lambda x: -x["strength"],
    )[:top_n]

    support = sorted(
        [{"level": c["mean"], "strength": len(c["values"])} for c in support_clusters],
        key=lambda x: -x["strength"],
    )[:top_n]

    return {"resistance": resistance, "support": support}


# ---------------------------------------------------------------------------
# Multi-Timeframe Confluence
# ---------------------------------------------------------------------------
def compute_confluence(mode: str, tf_data: dict) -> dict:
    """
    Compute a weighted confluence score across multiple timeframes.
    """
    if not tf_data:
        return {"score": 0.0, "alignment": "none", "bullish_tfs": 0, "bearish_tfs": 0}

    weights = TF_WEIGHTS.get(mode, {})

    score         = 0.0
    total_weight  = 0.0
    bullish_count = 0
    bearish_count = 0

    for tf, data in tf_data.items():
        weight = weights.get(tf, 0)
        total_weight += weight

        tf_score = data["score"]
        score   += tf_score * weight

        if tf_score > 0:
            bullish_count += 1
        elif tf_score < 0:
            bearish_count += 1

    normalised_score = round(score / total_weight, 3) if total_weight > 0 else 0.0

    n = len(tf_data)
    if n < 2:
        alignment = "single"
    elif bullish_count == n or bearish_count == n:
        alignment = "aligned"
    elif abs(bullish_count - bearish_count) <= 1:
        alignment = "mixed"
    else:
        alignment = "conflicting"

    return {
        "score":       normalised_score,
        "alignment":   alignment,
        "bullish_tfs": bullish_count,
        "bearish_tfs": bearish_count,
    }


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------
def detect_order_blocks(
    df: pd.DataFrame,
    threshold: float = 1.5,
    lookback: int = 5,
    top_n: int = 3,
) -> dict:
    """
    Identify bullish and bearish order blocks.

    An order block is the candle immediately before a strong impulsive move.
    Returns the last `top_n` blocks per type independently.
    """
    _validate_df(df, ["open", "high", "low", "close"], "detect_order_blocks")

    bullish_blocks: list[dict] = []
    bearish_blocks: list[dict] = []

    avg_moves = df["close"].diff().abs().rolling(20).mean()

    for i in range(lookback, len(df) - 1):
        move     = abs(df["close"].iloc[i + 1] - df["close"].iloc[i])
        avg_move = avg_moves.iloc[i]

        if pd.isna(avg_move) or avg_move == 0:
            continue

        if move > threshold * avg_move:
            candle = df.iloc[i]

            if candle["close"] < candle["open"]:
                bullish_blocks.append({
                    "type": "bullish",
                    "high": float(candle["high"]),
                    "low":  float(candle["low"]),
                })
            elif candle["close"] > candle["open"]:
                bearish_blocks.append({
                    "type": "bearish",
                    "high": float(candle["high"]),
                    "low":  float(candle["low"]),
                })

    return {
        "bullish": bullish_blocks[-top_n:],
        "bearish": bearish_blocks[-top_n:],
    }


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------
def compute_score(
    trend: dict,
    rsi:   dict,
    macd:  dict,
    atr:   dict,
    bb:    dict,
) -> float:
    """
    Compute a single directional score in [-1, 1].

    Weights
    -------
    Trend  : ±0.25
    RSI    : ±0.15
    MACD   : ±0.25
    ATR    : ±0.15  (expanding = conviction, contracting = caution)
    BB     : ±0.20  (percent_b position + squeeze penalty)
    Total  :  1.00

    BB Scoring Logic
    ----------------
    percent_b > 0.8  → bullish  (+0.10)
    percent_b < 0.2  → bearish  (-0.10)
    above_upper      → extra bullish momentum  (+0.10)
    below_lower      → extra bearish momentum  (-0.10)
    squeeze=True     → uncertainty penalty     (-0.10, capped, applied after)
    """
    score = 0.0

    # Trend (±0.25)
    if trend["direction"] == "UPTREND":
        score += 0.25
    elif trend["direction"] == "DOWNTREND":
        score -= 0.25

    # RSI (±0.15)
    if rsi["zone"] == "bullish":
        score += 0.15
    elif rsi["zone"] == "bearish":
        score -= 0.15

    # MACD (±0.25)
    if "bullish" in macd["signal"]:
        score += 0.25
    elif "bearish" in macd["signal"]:
        score -= 0.25

    # ATR (±0.15) — symmetric
    if atr["state"] == "expanding":
        score += 0.15
    elif atr["state"] == "contracting":
        score -= 0.15

    # BB (±0.20)
    percent_b   = bb["percent_b"]
    bb_position = bb["position"]
    bb_squeeze  = bb["squeeze"]

    # percent_b: directional lean based on where price sits in the bands
    if percent_b > 0.8:
        score += 0.10
    elif percent_b < 0.2:
        score -= 0.10

    # position: price outside bands = momentum confirmation
    if bb_position == "above_upper":
        score += 0.10
    elif bb_position == "below_lower":
        score -= 0.10

    # squeeze: bands compressed = direction not confirmed, reduce conviction
    if bb_squeeze:
        score -= 0.10

    return round(max(min(score, 1.0), -1.0), 2)


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------
def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect a liquidity sweep (stop-hunt) on the latest candle.

    Uses iloc[-2] for rolling reference to avoid look-ahead bias.
    """
    _validate_df(df, ["high", "low", "close"], "detect_liquidity_sweep")

    recent_high = df["high"].rolling(lookback).max().iloc[-2]
    recent_low  = df["low"].rolling(lookback).min().iloc[-2]

    high  = df["high"].iloc[-1]
    low   = df["low"].iloc[-1]
    close = df["close"].iloc[-1]

    sweep     = False
    direction: Optional[str]  = None
    level:    Optional[float] = None

    if high > recent_high and close < recent_high:
        sweep     = True
        direction = "bearish_reversal"
        level     = float(recent_high)

    elif low < recent_low and close > recent_low:
        sweep     = True
        direction = "bullish_reversal"
        level     = float(recent_low)

    return {"sweep": sweep, "direction": direction, "level": level}


# ---------------------------------------------------------------------------
# Breakout Detection
# ---------------------------------------------------------------------------
def detect_breakout(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Detect a confirmed price breakout on the latest candle.

    Uses iloc[-2] for rolling reference to avoid look-ahead bias.
    Strength is based on volume vs its 20-bar average.
    """
    _validate_df(df, ["high", "low", "close", "volume"], "detect_breakout")

    recent_high = df["high"].rolling(lookback).max().iloc[-2]
    recent_low  = df["low"].rolling(lookback).min().iloc[-2]

    close      = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    volume     = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    breakout  = False
    direction: Optional[str]  = None
    level:    Optional[float] = None
    strength: Optional[str]   = None

    if prev_close <= recent_high and close > recent_high:
        breakout  = True
        direction = "bullish"
        level     = float(recent_high)

    elif prev_close >= recent_low and close < recent_low:
        breakout  = True
        direction = "bearish"
        level     = float(recent_low)

    if breakout:
        strength = "strong" if volume > 1.5 * avg_volume else "moderate"

    return {
        "breakout":  breakout,
        "direction": direction,
        "level":     level,
        "strength":  strength,
    }


# ---------------------------------------------------------------------------
# BB Context  ← NEW
# ---------------------------------------------------------------------------
def detect_bb_context(df: pd.DataFrame) -> dict:
    """
    Enrich price action with Bollinger Band context on the latest candle.

    Returns
    -------
    dict with keys:
        band_touch   : str|None — "upper", "lower", or None
        inside_bands : bool     — price is between upper and lower bands
        squeeze      : bool     — bands are compressed vs their 20-bar mean width
        percent_b    : float    — normalised position within the bands (0–1)
    """
    _validate_df(df, ["close", "bb_upper", "bb_lower", "bb_middle"], "detect_bb_context")

    close  = df["close"].iloc[-1]
    upper  = df["bb_upper"].iloc[-1]
    lower  = df["bb_lower"].iloc[-1]

    band_width = upper - lower
    percent_b  = (close - lower) / band_width if band_width > 0 else 0.5

    # Tolerance: within 0.1% of band is considered a touch
    touch_tol  = band_width * 0.001
    band_touch: Optional[str] = None

    if close >= upper - touch_tol:
        band_touch = "upper"
    elif close <= lower + touch_tol:
        band_touch = "lower"

    inside_bands = lower < close < upper

    all_widths = df["bb_upper"] - df["bb_lower"]
    width_ma20 = all_widths.rolling(20).mean().iloc[-1]
    squeeze    = bool(band_width < width_ma20) if pd.notna(width_ma20) and width_ma20 > 0 else False

    return {
        "band_touch":   band_touch,
        "inside_bands": inside_bands,
        "squeeze":      squeeze,
        "percent_b":    round(percent_b, 3),
    }


# ---------------------------------------------------------------------------
# Price Action Enrichment
# ---------------------------------------------------------------------------
def enrich_price_action(df: pd.DataFrame) -> dict:
    return {
        "breakout":        detect_breakout(df),
        "liquidity_sweep": detect_liquidity_sweep(df),
        "bb_context":      detect_bb_context(df),    # ← added
    }
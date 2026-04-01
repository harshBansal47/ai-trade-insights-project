import numpy as np
import pandas as pd
from typing import Optional
from constants import TF_WEIGHTS


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
        Output of build_timeframe_features().

    Returns
    -------
    dict with keys:
        is_fake       : bool  — True if >= 2 directional conflicts exist
        conflicts     : list  — specific conflict labels
        low_volatility: bool  — True if ATR is contracting (separate flag,
                                not counted as a directional conflict)
    """
    trend      = tf_features["trend"]["direction"]
    rsi_zone   = tf_features["rsi"]["zone"]
    macd_signal = tf_features["macd"]["signal"]
    atr_state  = tf_features["atr"]["state"]

    conflicts: list[str] = []

    # RSI vs Trend conflicts
    if trend == "UPTREND" and rsi_zone == "overbought":
        conflicts.append("rsi_overbought_in_uptrend")

    if trend == "DOWNTREND" and rsi_zone == "oversold":
        conflicts.append("rsi_oversold_in_downtrend")

    # MACD conflicts
    if trend == "UPTREND" and "bearish" in macd_signal:
        conflicts.append("macd_bearish_against_trend")

    if trend == "DOWNTREND" and "bullish" in macd_signal:
        conflicts.append("macd_bullish_against_trend")

    # Low volatility is a separate risk flag — NOT a directional conflict.
    # Mixing it into `conflicts` caused a single real conflict + low ATR
    # to incorrectly flag a signal as fake.
    is_low_vol = atr_state == "contracting"

    # A signal is fake if there are >= 2 directional conflicts,
    # OR >= 1 directional conflict combined with low volatility.
    is_fake = len(conflicts) >= 2 or (len(conflicts) >= 1 and is_low_vol)

    return {
        "is_fake": is_fake,
        "conflicts": conflicts,
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

    Parameters
    ----------
    df     : DataFrame with 'high' and 'low' columns.
    window : Number of candles on each side to compare.

    Returns
    -------
    (highs, lows) — lists of price levels.
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

    Levels are sorted first to ensure order-independent assignment —
    unsorted input caused a level to attach to the wrong cluster when
    an earlier level shifted the cluster mean away from it.

    Parameters
    ----------
    levels    : List of raw price levels (highs or lows).
    tolerance : Max relative distance to merge into an existing cluster.

    Returns
    -------
    List of cluster dicts: {"mean": float, "values": list[float]}
    """
    if not levels:
        return []

    clusters: list[dict] = []

    # Sort so nearby levels are evaluated adjacently — prevents order
    # dependency in cluster assignment.
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

    Parameters
    ----------
    df    : OHLCV DataFrame.
    top_n : Number of strongest zones to return per side.

    Returns
    -------
    dict with keys 'resistance' and 'support', each a list of
    {"level": float, "strength": int} sorted by descending strength.
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

    Parameters
    ----------
    mode    : Trading mode key used to look up TF_WEIGHTS (e.g. "scalp").
    tf_data : Dict of {timeframe: {"score": float, ...}}.

    Returns
    -------
    dict with keys: score, alignment, bullish_tfs, bearish_tfs.

    Notes
    -----
    - score is normalised by total_weight to keep it in [-1, 1].
    - alignment requires >= 2 timeframes to be meaningful.
    """
    if not tf_data:
        return {"score": 0.0, "alignment": "none", "bullish_tfs": 0, "bearish_tfs": 0}

    weights = TF_WEIGHTS.get(mode, {})

    score        = 0.0
    total_weight = 0.0
    bullish_count = 0
    bearish_count = 0

    for tf, data in tf_data.items():
        weight = weights.get(tf, 0)
        total_weight += weight

        tf_score = data["score"]
        score += tf_score * weight

        if tf_score > 0:
            bullish_count += 1
        elif tf_score < 0:
            bearish_count += 1

    # Normalise — avoids inflated scores when weights don't sum to 1.
    # Guard against division by zero when mode is unknown / weights missing.
    normalised_score = round(score / total_weight, 3) if total_weight > 0 else 0.0

    n = len(tf_data)
    if n < 2:
        # A single timeframe cannot be "aligned" — it's just one reading.
        alignment = "single"
    elif bullish_count == n or bearish_count == n:
        alignment = "aligned"
    elif abs(bullish_count - bearish_count) <= 1:
        alignment = "mixed"
    else:
        alignment = "conflicting"

    return {
        "score": normalised_score,
        "alignment": alignment,
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
    The previous implementation returned only `blocks[-3:]` across both
    types, meaning all 3 results could be the same type.  This version
    returns the last `top_n` blocks per type independently.

    Parameters
    ----------
    df        : OHLCV DataFrame.
    threshold : Move must exceed `threshold × avg_move` to qualify.
    lookback  : Min index offset from start to begin evaluating.
    top_n     : Max order blocks to return per type.

    Returns
    -------
    dict with keys 'bullish' and 'bearish', each a list of
    {"type", "high", "low"} dicts (most recent last).
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

            # Bullish OB: bearish candle immediately before a strong up move
            if candle["close"] < candle["open"]:
                bullish_blocks.append({
                    "type": "bullish",
                    "high": float(candle["high"]),
                    "low":  float(candle["low"]),
                })

            # Bearish OB: bullish candle immediately before a strong down move
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
) -> float:
    """
    Compute a single directional score in [-1, 1].

    Weights
    -------
    Trend   : ±0.30
    RSI     : ±0.20
    MACD    : ±0.30
    ATR     : ±0.20  (expanding = conviction boost, contracting = penalty)
    Total   :  1.00

    The ATR component is now symmetric — previously expanding gave +0.2
    but contracting gave 0, making the score asymmetric.
    """
    score = 0.0

    # Trend (±0.30)
    if trend["direction"] == "UPTREND":
        score += 0.30
    elif trend["direction"] == "DOWNTREND":
        score -= 0.30

    # RSI (±0.20)
    if rsi["zone"] == "bullish":
        score += 0.20
    elif rsi["zone"] == "bearish":
        score -= 0.20

    # MACD (±0.30)
    if "bullish" in macd["signal"]:
        score += 0.30
    elif "bearish" in macd["signal"]:
        score -= 0.30

    # ATR (±0.20) — symmetric: expansion = conviction, contraction = caution
    if atr["state"] == "expanding":
        score += 0.20
    elif atr["state"] == "contracting":
        score -= 0.20

    return round(max(min(score, 1.0), -1.0), 2)


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------
def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect a liquidity sweep (stop-hunt) on the latest candle.

    A sweep above the recent high that closes back below it signals a
    bearish reversal.  A sweep below the recent low that closes back
    above it signals a bullish reversal.

    Uses iloc[-2] for the rolling window reference to avoid look-ahead
    bias — the current candle's high/low must not influence the level.

    Returns
    -------
    dict with keys: sweep (bool), direction (str|None), level (float|None).
    """
    _validate_df(df, ["high", "low", "close"], "detect_liquidity_sweep")

    # Avoid look-ahead: reference level is calculated excluding the last bar
    recent_high = df["high"].rolling(lookback).max().iloc[-2]
    recent_low  = df["low"].rolling(lookback).min().iloc[-2]

    high  = df["high"].iloc[-1]
    low   = df["low"].iloc[-1]
    close = df["close"].iloc[-1]

    sweep    = False
    direction: Optional[str]   = None
    level:    Optional[float]  = None

    # Wick above recent high but closes back below → bearish reversal
    if high > recent_high and close < recent_high:
        sweep     = True
        direction = "bearish_reversal"
        level     = float(recent_high)

    # Wick below recent low but closes back above → bullish reversal
    elif low < recent_low and close > recent_low:
        sweep     = True
        direction = "bullish_reversal"
        level     = float(recent_low)

    return {
        "sweep":     sweep,
        "direction": direction,
        # level is None when no sweep occurred — previously always returned
        # a price even when sweep=False (str(None) bug).
        "level":     level,
    }


# ---------------------------------------------------------------------------
# Breakout Detection
# ---------------------------------------------------------------------------
def detect_breakout(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Detect a confirmed price breakout on the latest candle.

    Strength is based on volume vs its 20-bar average.
    Uses iloc[-2] for rolling reference to avoid look-ahead bias.

    Returns
    -------
    dict with keys: breakout (bool), direction (str|None),
                    level (float|None), strength (str|None).

    Notes
    -----
    level and strength are None when breakout=False — previously they
    always returned a stale price level and "weak" even with no breakout.
    """
    _validate_df(df, ["high", "low", "close", "volume"], "detect_breakout")

    # Avoid look-ahead bias
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

    # Bullish breakout: previous close was inside range, now above
    if prev_close <= recent_high and close > recent_high:
        breakout  = True
        direction = "bullish"
        level     = float(recent_high)

    # Bearish breakout: previous close was inside range, now below
    elif prev_close >= recent_low and close < recent_low:
        breakout  = True
        direction = "bearish"
        level     = float(recent_low)

    # Strength is only meaningful when a breakout actually occurred
    if breakout:
        strength = "strong" if volume > 1.5 * avg_volume else "moderate"

    return {
        "breakout":  breakout,
        "direction": direction,
        "level":     level,
        # strength=None (not "weak") when breakout=False — avoids misleading
        # downstream consumers into thinking a non-event has a strength rating.
        "strength":  strength,
    }


# ---------------------------------------------------------------------------
# Price Action Enrichment
# ---------------------------------------------------------------------------
def enrich_price_action(df: pd.DataFrame) -> dict:
    return {
        "breakout":        detect_breakout(df),
        "liquidity_sweep": detect_liquidity_sweep(df),
    }
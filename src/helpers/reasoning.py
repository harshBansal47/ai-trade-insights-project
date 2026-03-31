

import numpy as np

from constants import TF_WEIGHTS


def detect_fake_signal(tf_features):
    """
    tf_features = output of build_timeframe_features()
    """

    trend = tf_features["trend"]["direction"]
    rsi_zone = tf_features["rsi"]["zone"]
    macd_signal = tf_features["macd"]["signal"]
    atr_state = tf_features["atr"]["state"]

    conflicts = []

    # RSI vs Trend conflict
    if trend == "UPTREND" and rsi_zone == "overbought":
        conflicts.append("rsi_overbought_in_uptrend")

    if trend == "DOWNTREND" and rsi_zone == "oversold":
        conflicts.append("rsi_oversold_in_downtrend")

    # MACD conflict
    if trend == "UPTREND" and "bearish" in macd_signal:
        conflicts.append("macd_bearish_against_trend")

    if trend == "DOWNTREND" and "bullish" in macd_signal:
        conflicts.append("macd_bullish_against_trend")

    # Low volatility trap
    if atr_state == "contracting":
        conflicts.append("low_volatility")

    is_fake = len(conflicts) >= 2

    return {
        "is_fake": is_fake,
        "conflicts": conflicts
    }

def find_swing_points(df, window=5):
    highs = []
    lows = []

    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == max(df["high"].iloc[i-window:i+window]):
            highs.append(df["high"].iloc[i])

        if df["low"].iloc[i] == min(df["low"].iloc[i-window:i+window]):
            lows.append(df["low"].iloc[i])

    return highs, lows


def cluster_levels(levels, tolerance=0.005):
    clusters = []

    for level in levels:
        placed = False
        for cluster in clusters:
            if abs(level - cluster["mean"]) / cluster["mean"] < tolerance:
                cluster["values"].append(level)
                cluster["mean"] = np.mean(cluster["values"])
                placed = True
                break

        if not placed:
            clusters.append({
                "mean": level,
                "values": [level]
            })

    return clusters


def get_sr_zones(df):
    highs, lows = find_swing_points(df)

    resistance_clusters = cluster_levels(highs)
    support_clusters = cluster_levels(lows)

    resistance = sorted(
        [{"level": c["mean"], "strength": len(c["values"])} for c in resistance_clusters],
        key=lambda x: -x["strength"]
    )[:3]

    support = sorted(
        [{"level": c["mean"], "strength": len(c["values"])} for c in support_clusters],
        key=lambda x: -x["strength"]
    )[:3]

    return {
        "resistance": resistance,
        "support": support
    }





def compute_confluence(mode, tf_data):
    weights = TF_WEIGHTS.get(mode, {})

    score = 0
    total_weight = 0

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

    alignment = (
        "aligned" if bullish_count == len(tf_data) or bearish_count == len(tf_data)
        else "mixed" if abs(bullish_count - bearish_count) <= 1
        else "conflicting"
    )

    return {
        "score": round(score, 3),
        "alignment": alignment,
        "bullish_tfs": bullish_count,
        "bearish_tfs": bearish_count
    }

def detect_order_blocks(df, threshold=1.5):
    blocks = []

    for i in range(5, len(df)-1):
        move = abs(df["close"].iloc[i+1] - df["close"].iloc[i])
        avg_move = df["close"].diff().abs().rolling(20).mean().iloc[i]

        if move > threshold * avg_move:
            candle = df.iloc[i]

            # Bullish OB (down candle before up move)
            if candle["close"] < candle["open"]:
                blocks.append({
                    "type": "bullish",
                    "high": candle["high"],
                    "low": candle["low"]
                })

            # Bearish OB
            elif candle["close"] > candle["open"]:
                blocks.append({
                    "type": "bearish",
                    "high": candle["high"],
                    "low": candle["low"]
                })

    return blocks[-3:] 



# ---------------------------
# SCORE (VERY IMPORTANT)
# ---------------------------
def compute_score(trend, rsi, macd, atr):
    score = 0

    if trend["direction"] == "UPTREND":
        score += 0.3
    elif trend["direction"] == "DOWNTREND":
        score -= 0.3

    if rsi["zone"] == "bullish":
        score += 0.2
    elif rsi["zone"] == "bearish":
        score -= 0.2

    if "bullish" in macd["signal"]:
        score += 0.3
    elif "bearish" in macd["signal"]:
        score -= 0.3

    if atr["state"] == "expanding":
        score += 0.2

    return round(max(min(score, 1), -1), 2)

def detect_liquidity_sweep(df, lookback=20):
    recent_high = df["high"].rolling(lookback).max().iloc[-2]
    recent_low = df["low"].rolling(lookback).min().iloc[-2]

    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    close = df["close"].iloc[-1]

    sweep = False
    direction = None

    # Sweep above high (fake breakout → bearish)
    if high > recent_high and close < recent_high:
        sweep = True
        direction = "bearish_reversal"

    # Sweep below low (fake breakdown → bullish)
    elif low < recent_low and close > recent_low:
        sweep = True
        direction = "bullish_reversal"

    return {
        "sweep": sweep,
        "direction": direction,
        "level": float(recent_high if "bearish" in str(direction) else recent_low)
    }

def detect_breakout(df, lookback=50):
    recent_high = df["high"].rolling(lookback).max().iloc[-2]
    recent_low = df["low"].rolling(lookback).min().iloc[-2]

    close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]

    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    breakout = False
    direction = None
    strength = "weak"

    # Bullish breakout
    if prev_close <= recent_high and close > recent_high:
        breakout = True
        direction = "bullish"

    # Bearish breakout
    elif prev_close >= recent_low and close < recent_low:
        breakout = True
        direction = "bearish"

    # Strength check
    if breakout:
        if volume > 1.5 * avg_volume:
            strength = "strong"
        else:
            strength = "moderate"

    return {
        "breakout": breakout,
        "direction": direction,
        "level": float(recent_high if direction == "bullish" else recent_low),
        "strength": strength
    }


def enrich_price_action(df):
    return {
        "breakout": detect_breakout(df),
        "liquidity_sweep": detect_liquidity_sweep(df)
    }

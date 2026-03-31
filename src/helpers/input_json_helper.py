import numpy as np

from helpers.indicator_features import get_atr_features, get_macd_features, get_rsi_features, get_trend
from helpers.input_payload_calculation_pipeline import TF_WEIGHTS
from helpers.reasoning import compute_confluence, compute_score, detect_order_blocks, enrich_price_action, get_sr_zones





# ---------------------------
# TIMEFRAME BUILDER
# ---------------------------
def build_timeframe_features(df, mode):
    trend = get_trend(df)
    rsi = get_rsi_features(df)
    macd = get_macd_features(df)
    atr = get_atr_features(df)

    score = compute_score(trend, rsi, macd, atr)

    sr = get_sr_zones(df)
    order_blocks = detect_order_blocks(df)

    price_action = enrich_price_action(df)

    return {
        "trend": trend,
        "rsi": rsi,
        "macd": macd,
        "atr": atr,
        "sr_zones": sr,
        "order_blocks": order_blocks,
        "price_action": price_action,
        "score": score
    }



# ---------------------------
# GLOBAL SIGNALS
# ---------------------------
def build_global_signals(tf_data):
    bullish = sum(1 for tf in tf_data.values() if tf["score"] > 0)
    total = len(tf_data)

    alignment = f"{bullish}/{total} bullish"

    confidence = round(sum(tf["score"] for tf in tf_data.values()) / total, 2)
    confluence = compute_confluence(mode, tf_data)


    return {
        "volume_spike": False,  # plug your logic
        "volatility_regime": "moderate",
        "recent_breakout": False,
        "trend_alignment": alignment,
        "confidence_score": confidence,

        "trend_alignment": confluence["alignment"],
        "confidence_score": confluence["score"],
        "bullish_tfs": confluence["bullish_tfs"],
        "bearish_tfs": confluence["bearish_tfs"]
    }



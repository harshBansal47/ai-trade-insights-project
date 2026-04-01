from enum import Enum
from typing import Optional

import redis
from core.celery import celery
from helpers.indicator_features import get_atr_features, get_macd_features, get_rsi_features, get_trend
from helpers.reasoning import compute_confluence, compute_score, detect_fake_signal, detect_order_blocks, enrich_price_action, get_sr_zones
from src.core.redis import app_redis_client
from constants import MODE_TIMEFRAMES
from helpers.data_loader import process_symbol_multi_timeframe


class Mode(Enum):
    SCALPER = "SCALPER"
    SWING = "SWING"
    POSITION = "POSITION"



async def run_analysis(symbol: str, mode: Mode):
    print(f"Analyzing {symbol} in {mode.value} mode...")
    mode_timeframes = MODE_TIMEFRAMES[mode.value]
    print(f"Timeframes for {mode.value} mode: {mode_timeframes}")
    input_report = prepare_input_report.delay(
        symbol=symbol,  
        timeframes=mode_timeframes,
        mode_value=mode.value,
        redis_client=app_redis_client
    )
    result = input_report.get(timeout=30)
    if result is None:  
        print(f"Failed to prepare input report for {symbol} in {mode.value} mode.")
    else:
        print(f"Input report for {symbol} in {mode.value} mode: {result}")      



@celery.task(
    bind=True,
    name="tasks.prepare_input_report",
    max_retries=3,
    default_retry_delay=10,   
)
def prepare_input_report(
    self,
    symbol:     str,
    timeframes: list[str],
    mode_value: str,
    redis_client:redis.Redis,
) -> Optional[dict]:
    
    try:
        raw_data = process_symbol_multi_timeframe(
            symbol=symbol,
            timeframes=timeframes,
            redis_client=redis_client
        )
    except Exception as exc:
        raise self.retry(exc=exc)
 
    report = assemble_report(symbol, mode_value, raw_data)
 
    if report is None:
        return None
 
    return report




def assemble_report(
    symbol: str,
    mode:   Mode,
    raw_data: list[dict],
) -> Optional[dict]:
    tf_features: dict[str, dict] = {}
    for result in raw_data:
        tf = result["timeframe"]
 
        if result["status"] != "success" or result.get("rows", 0) == 0:
            continue

        try:
            tf_features[tf] = build_timeframe_features(result["df"], mode)
        except Exception as exc:
            return None
 
    if not tf_features:
        return None
 
    global_signals = build_global_signals(mode, tf_features)
 
    return {
        "symbol":         symbol,
        "mode":           mode.value,
        "timeframes":     tf_features,
        "global_signals": global_signals,
    }


# ---------------------------------------------------------------------------
# TIMEFRAME BUILDER
# ---------------------------------------------------------------------------
def build_timeframe_features(df, mode: str) -> dict:
    """
    Build the full feature set for a single timeframe DataFrame.
 
    Parameters
    ----------
    df   : OHLCV DataFrame for one timeframe.
    mode : Trading mode string (e.g. "scalp", "swing").
 
    Returns
    -------
    Feature dict containing trend, rsi, macd, atr, sr_zones,
    order_blocks, price_action, fake_signal, and score.
    """
    trend = get_trend(df)
    rsi   = get_rsi_features(df)
    macd  = get_macd_features(df)
    atr   = get_atr_features(df)
 
    score = compute_score(trend, rsi, macd, atr)
 
    # Build the indicator sub-dict first so detect_fake_signal can consume it
    tf_features = {
        "trend": trend,
        "rsi":   rsi,
        "macd":  macd,
        "atr":   atr,
    }
 
    return {
        **tf_features,
        "sr_zones":     get_sr_zones(df),
        "order_blocks": detect_order_blocks(df),
        "price_action": enrich_price_action(df),
        "fake_signal":  detect_fake_signal(tf_features), 
        "score":        score,
    }
 
 
# ---------------------------------------------------------------------------
# GLOBAL SIGNALS
# ---------------------------------------------------------------------------
def build_global_signals(mode: str, tf_data: dict) -> dict:
    """
    Aggregate per-timeframe features into a single global signal payload.
 
    Parameters
    ----------
    mode    : Trading mode string — required by compute_confluence for
              TF weight lookup. Was missing as a parameter in the original,
              causing a NameError at runtime.
    tf_data : Dict of {timeframe_str: build_timeframe_features() output}.
 
    Returns
    -------
    Global signal dict.
    """
    if not tf_data:
        return {
            "volume_spike":      False,
            "volatility_regime": "unknown",
            "recent_breakout":   False,
            "trend_alignment":   "none",
            "confidence_score":  0.0,
            "bullish_tfs":       0,
            "bearish_tfs":       0,
        }
 
    # confluence is the single source of truth for alignment and score.
    # The original computed `alignment` and `confidence` manually and then
    # immediately overwrote them with confluence values via duplicate dict
    # keys — making those two lines completely dead code.
    confluence = compute_confluence(mode, tf_data)
 
    return {
        "volume_spike":      False,        # plug your volume-spike logic here
        "volatility_regime": "moderate",   # plug your regime classifier here
        "recent_breakout":   False,        # plug your breakout logic here
        "trend_alignment":   confluence["alignment"],
        "confidence_score":  confluence["score"],
        "bullish_tfs":       confluence["bullish_tfs"],
        "bearish_tfs":       confluence["bearish_tfs"],
    }
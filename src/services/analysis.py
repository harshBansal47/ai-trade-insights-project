from enum import Enum
from typing import Optional
import redis

from src.core.celery import celery_app as celery
from src.core.logger import setup_logger

from src.helpers.indicator_features import (
    get_atr_features,
    get_macd_features,
    get_rsi_features,
    get_trend,
)

from src.helpers.reasoning import (
    compute_confluence,
    compute_score,
    detect_fake_signal,
    detect_order_blocks,
    enrich_price_action,
    get_sr_zones,
)

from src.constants import MODE_TIMEFRAMES
from src.helpers.data_loader import process_symbol_multi_timeframe

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# MODE ENUM
# ---------------------------------------------------------------------------
class Mode(Enum):
    SCALPER = "SCALPER"
    SWING = "SWING"
    POSITION = "POSITION"


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
async def run_analysis(symbol: str, mode: Mode):
    logger.info(f"[RUN ANALYSIS] symbol={symbol} mode={mode.value}")

    try:
        mode_timeframes = MODE_TIMEFRAMES[mode.value]
        logger.info(f"[TIMEFRAMES] {symbol} -> {mode_timeframes}")

        task = prepare_input_report.delay(
            symbol=symbol,
            timeframes=mode_timeframes,
            mode_value=mode.value,
        )

        logger.info(f"[TASK SUBMITTED] id={task.id} symbol={symbol}")

        return {"task_id": task.id, "status": "submitted"}

    except Exception as e:
        logger.error(
            f"[RUN ANALYSIS ERROR] symbol={symbol} mode={mode.value} error={e}",
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# CELERY TASK
# ---------------------------------------------------------------------------
@celery.task(
    bind=True,
    name="tasks.prepare_input_report",
    max_retries=3,
    default_retry_delay=10,
)
def prepare_input_report(
    self, symbol: str, timeframes: list[str], mode_value: str
) -> Optional[dict]:

    logger.info(
        f"[TASK START] id={self.request.id} symbol={symbol} mode={mode_value}"
    )

    try:
        raw_data = process_symbol_multi_timeframe(
            symbol=symbol, timeframes=timeframes
        )

        logger.info(
            f"[DATA LOADED] symbol={symbol} tf_count={len(raw_data)}"
        )

    except Exception as exc:
        logger.error(
            f"[TASK ERROR - DATA LOAD] symbol={symbol} error={exc}",
            exc_info=True,
        )
        raise self.retry(exc=exc)

    try:
        report = assemble_report(symbol, mode_value, raw_data)

        if report is None:
            logger.warning(f"[EMPTY REPORT] symbol={symbol}")
            return None

        logger.info(f"[TASK SUCCESS] symbol={symbol}")

        return report

    except Exception as exc:
        logger.error(
            f"[TASK ERROR - REPORT BUILD] symbol={symbol} error={exc}",
            exc_info=True,
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# REPORT ASSEMBLY
# ---------------------------------------------------------------------------
def assemble_report(
    symbol: str,
    mode: str,
    raw_data: list[dict],
) -> Optional[dict]:

    logger.info(f"[ASSEMBLE REPORT] symbol={symbol}")

    tf_features: dict[str, dict] = {}

    for result in raw_data:
        tf = result["timeframe"]

        if result["status"] != "success" or result.get("rows", 0) == 0:
            logger.warning(f"[SKIP TF] {symbol} | {tf}")
            continue

        try:
            logger.info(f"[BUILD TF FEATURES] {symbol} | {tf}")

            tf_features[tf] = build_timeframe_features(result["df"], mode)

        except Exception as exc:
            logger.error(
                f"[TF BUILD ERROR] {symbol} | {tf} | {exc}",
                exc_info=True,
            )
            return None

    if not tf_features:
        logger.warning(f"[NO VALID TF FEATURES] symbol={symbol}")
        return None

    global_signals = build_global_signals(mode, tf_features)

    logger.info(f"[REPORT READY] symbol={symbol}")

    return {
        "symbol": symbol,
        "mode": mode,
        "timeframes": tf_features,
        "global_signals": global_signals,
    }


# ---------------------------------------------------------------------------
# TIMEFRAME BUILDER
# ---------------------------------------------------------------------------
def build_timeframe_features(df, mode: str) -> dict:
    logger.debug("[TF FEATURE START]")

    try:
        trend = get_trend(df)
        rsi = get_rsi_features(df)
        macd = get_macd_features(df)
        atr = get_atr_features(df)

        score = compute_score(trend, rsi, macd, atr)

        tf_features = {
            "trend": trend,
            "rsi": rsi,
            "macd": macd,
            "atr": atr,
        }

        result = {
            **tf_features,
            "sr_zones": get_sr_zones(df),
            "order_blocks": detect_order_blocks(df),
            "price_action": enrich_price_action(df),
            "fake_signal": detect_fake_signal(tf_features),
            "score": score,
        }

        logger.debug("[TF FEATURE SUCCESS]")
        return result

    except Exception as e:
        logger.error(f"[TF FEATURE ERROR] {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# GLOBAL SIGNALS
# ---------------------------------------------------------------------------
def build_global_signals(mode: str, tf_data: dict) -> dict:
    logger.info("[GLOBAL SIGNALS BUILD]")

    if not tf_data:
        logger.warning("[GLOBAL SIGNALS EMPTY INPUT]")
        return {
            "volume_spike": False,
            "volatility_regime": "unknown",
            "recent_breakout": False,
            "trend_alignment": "none",
            "confidence_score": 0.0,
            "bullish_tfs": 0,
            "bearish_tfs": 0,
        }

    try:
        confluence = compute_confluence(mode, tf_data)

        result = {
            "volume_spike": False,
            "volatility_regime": "moderate",
            "recent_breakout": False,
            "trend_alignment": confluence["alignment"],
            "confidence_score": confluence["score"],
            "bullish_tfs": confluence["bullish_tfs"],
            "bearish_tfs": confluence["bearish_tfs"],
        }

        logger.info("[GLOBAL SIGNALS SUCCESS]")
        return result

    except Exception as e:
        logger.error(f"[GLOBAL SIGNAL ERROR] {e}", exc_info=True)
        raise
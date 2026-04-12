from enum import Enum
from typing import Optional

from src.core.celery import celery_app as celery
from src.core.logger import setup_logger
from src.core.database import SyncSessionLocal  # you need a sync session for Celery

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
from src.models.task import Task, TaskStatus

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# MODE ENUM
# ---------------------------------------------------------------------------
class Mode(Enum):
    SCALPER = "SCALPER"
    SWING = "SWING"
    POSITION = "POSITION"


# ---------------------------------------------------------------------------
# CELERY TASK  — signature now matches exactly what the router sends
# ---------------------------------------------------------------------------
@celery.task(
    bind=True,
    name="tasks.run_analysis",
    max_retries=3,
    default_retry_delay=10,
)
def run_analysis(
    self,
    task_id: str,
    user_id: str,
    coin: str,
    coin_symbol: str,
    mode_value: str,
    message: str,
) -> Optional[dict]:

    logger.info(
        f"[TASK START] id={self.request.id} task_id={task_id} "
        f"symbol={coin_symbol} mode={mode_value}"
    )

    # ── Mark task as PROCESSING ──────────────────────────────────────────
    _update_task(task_id, status=TaskStatus.PROCESSING)

    # ── Load market data ─────────────────────────────────────────────────
    try:
        mode_timeframes = MODE_TIMEFRAMES[mode_value]

        raw_data = process_symbol_multi_timeframe(
            symbol=coin_symbol, timeframes=mode_timeframes
        )

        logger.info(
            f"[DATA LOADED] symbol={coin_symbol} tf_count={len(raw_data)}"
        )

    except Exception as exc:
        logger.error(
            f"[TASK ERROR - DATA LOAD] symbol={coin_symbol} error={exc}",
            exc_info=True,
        )
        _update_task(task_id, status=TaskStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc)

    # ── Build report ─────────────────────────────────────────────────────
    try:
        report = assemble_report(
            symbol=coin_symbol,
            mode=mode_value,
            raw_data=raw_data,
        )

        if report is None:
            logger.warning(f"[EMPTY REPORT] symbol={coin_symbol}")
            _update_task(
                task_id,
                status=TaskStatus.FAILED,
                error="Analysis returned no data.",
            )
            return None

        # ── Save result and mark COMPLETED ────────────────────────────
        _update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result=report,
        )

        logger.info(f"[TASK SUCCESS] task_id={task_id} symbol={coin_symbol}")
        return report

    except Exception as exc:
        logger.error(
            f"[TASK ERROR - REPORT BUILD] symbol={coin_symbol} error={exc}",
            exc_info=True,
        )
        _update_task(task_id, status=TaskStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# DB HELPER — sync because Celery workers are synchronous
# ---------------------------------------------------------------------------
def _update_task(
    task_id: str,
    status: TaskStatus,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Open a short-lived sync session, update the Task row, and close."""
    from datetime import datetime, timezone

    try:
        with SyncSessionLocal() as db:
            task = db.get(Task, task_id)

            if not task:
                logger.warning(f"[UPDATE TASK] task_id={task_id} not found")
                return

            task.status = status

            if result is not None:
                task.result = result

            if error is not None:
                task.error = error

            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.now(timezone.utc)

            db.add(task)
            db.commit()

            logger.info(f"[TASK UPDATED] task_id={task_id} status={status}")

    except Exception as e:
        logger.error(f"[DB UPDATE ERROR] task_id={task_id} error={e}", exc_info=True)


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
        rsi   = get_rsi_features(df)
        macd  = get_macd_features(df)
        atr   = get_atr_features(df)

        score = compute_score(trend, rsi, macd, atr)

        tf_features = {
            "trend": trend,
            "rsi":   rsi,
            "macd":  macd,
            "atr":   atr,
        }

        result = {
            **tf_features,
            "sr_zones":     get_sr_zones(df),
            "order_blocks": detect_order_blocks(df),
            "price_action": enrich_price_action(df),
            "fake_signal":  detect_fake_signal(tf_features),
            "score":        score,
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
            "volume_spike":       False,
            "volatility_regime":  "unknown",
            "recent_breakout":    False,
            "trend_alignment":    "none",
            "confidence_score":   0.0,
            "bullish_tfs":        0,
            "bearish_tfs":        0,
        }

    try:
        confluence = compute_confluence(mode, tf_data)

        result = {
            "volume_spike":      False,
            "volatility_regime": "moderate",
            "recent_breakout":   False,
            "trend_alignment":   confluence["alignment"],
            "confidence_score":  confluence["score"],
            "bullish_tfs":       confluence["bullish_tfs"],
            "bearish_tfs":       confluence["bearish_tfs"],
        }

        logger.info("[GLOBAL SIGNALS SUCCESS]")
        return result

    except Exception as e:
        logger.error(f"[GLOBAL SIGNAL ERROR] {e}", exc_info=True)
        raise
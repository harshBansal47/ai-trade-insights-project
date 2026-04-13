# src/agents/chain.py

"""
Signal Chain
------------
Orchestrates the full pipeline:

  raw_data
    └─► assemble_report()
          └─► memory context injection
                └─► Gemini LLM (via crypto_analysis_prompt)
                      └─► parse_signal_output()
                            └─► save_signal() → memory
                                  └─► SignalOutput

Retry logic:
  - Up to MAX_RETRIES attempts on LLM / parse failure
  - Exponential back-off between retries
  - Returns a safe fallback SignalOutput on total failure

Usage
-----
    chain  = SignalChain()
    signal = chain.run(report)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI



from agentic_workflow.memory import build_memory_context, save_signal
from agentic_workflow.parser import parse_signal_output
from agentic_workflow.schemas import RiskLevel, SignalOutput, SignalType
from src.core.config     import settings
from src.core.logger     import setup_logger
from src.agentic_workflow.prompts import crypto_analysis_prompt

logger = setup_logger(__name__)

MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 2.0   # seconds — doubles each retry


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class SignalChain:
    """
    End-to-end chain: report dict → validated SignalOutput.

    Parameters
    ----------
    temperature : LLM temperature (lower = more deterministic signals)
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=False,
        )
        logger.info("[CHAIN] SignalChain initialised")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, report: dict) -> SignalOutput:
        """
        Run the full pipeline on a report dict.

        Parameters
        ----------
        report : Output of assemble_report() — must have symbol, mode,
                 timeframes, global_signals keys.

        Returns
        -------
        SignalOutput — always returns something (fallback on total failure).
        """
        symbol = report.get("symbol", "UNKNOWN")
        mode   = report.get("mode",   "UNKNOWN")

        logger.info(f"[CHAIN] Starting pipeline for {symbol}|{mode}")

        # Inject memory context into the report so the LLM can see history
        report = self._inject_memory(report, symbol, mode)

        # Attempt LLM call with retries
        signal = self._run_with_retries(report, symbol, mode)

        # Persist to memory
        try:
            save_signal(symbol, mode, signal.model_dump())
        except Exception as exc:
            logger.warning(f"[CHAIN] Memory save failed (non-fatal): {exc}")

        logger.info(
            f"[CHAIN] Completed {symbol}|{mode} → "
            f"signal={signal.signal} confidence={signal.confidence}"
        )
        return signal

    def run_safe(self, report: dict) -> Optional[SignalOutput]:
        """Same as run() but returns None on unrecoverable failure."""
        try:
            return self.run(report)
        except Exception as exc:
            logger.error(f"[CHAIN] run_safe caught: {exc}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Private — retry loop
    # ------------------------------------------------------------------

    def _run_with_retries(
        self,
        report: dict,
        symbol: str,
        mode:   str,
    ) -> SignalOutput:
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(f"[CHAIN] Attempt {attempt}/{MAX_RETRIES} for {symbol}")
                signal = self._invoke(report)

                if signal is not None:
                    return signal

                raise RuntimeError("Parser returned None")

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[CHAIN] Attempt {attempt} failed for {symbol}: {exc}"
                )
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.debug(f"[CHAIN] Retrying in {delay:.1f}s")
                    time.sleep(delay)

        logger.error(
            f"[CHAIN] All {MAX_RETRIES} attempts failed for {symbol}. "
            f"Returning fallback signal. Last error: {last_exc}"
        )
        return _build_fallback_signal(report)

    # ------------------------------------------------------------------
    # Private — single LLM invoke + parse
    # ------------------------------------------------------------------

    def _invoke(self, report: dict) -> Optional[SignalOutput]:
        """Format prompt → call LLM → parse response."""
        report_json = json.dumps(report, indent=2, default=str)

        messages = crypto_analysis_prompt.format_messages(
            input_data=report_json,
            symbol=report.get("symbol", "UNKNOWN"),
            mode=report.get("mode",   "UNKNOWN"),
        )

        logger.debug("[CHAIN] Invoking LLM")
        raw = self._llm.invoke(messages)

        return parse_signal_output(raw, report)

    # ------------------------------------------------------------------
    # Private — memory injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_memory(report: dict, symbol: str, mode: str) -> dict:
        """
        Append historical signal context to the report under
        'memory_context' key so the LLM prompt can reference it.
        """
        try:
            context = build_memory_context(symbol, mode, limit=3)
            if context:
                report = {**report, "memory_context": context}
                logger.debug(f"[CHAIN] Injected memory context for {symbol}|{mode}")
        except Exception as exc:
            logger.warning(f"[CHAIN] Memory inject failed (non-fatal): {exc}")
        return report


# ---------------------------------------------------------------------------
# Fallback signal — returned when all retries are exhausted
# ---------------------------------------------------------------------------

def _build_fallback_signal(report: dict) -> SignalOutput:
    """
    Construct a safe, minimal SignalOutput when the LLM pipeline fails
    completely. Always returns NO_TRADE with maximum risk.
    """
    symbol = report.get("symbol", "UNKNOWN")
    mode   = report.get("mode",   "UNKNOWN")

    # Try to extract dominant trend from the report for the fallback
    dominant_trend = "SIDEWAYS"
    try:
        tfs = report.get("timeframes", {})
        directions = [
            tfs[tf]["trend"]["direction"]
            for tf in tfs
            if "trend" in tfs.get(tf, {})
        ]
        if directions.count("UPTREND")   > len(directions) // 2:
            dominant_trend = "UPTREND"
        elif directions.count("DOWNTREND") > len(directions) // 2:
            dominant_trend = "DOWNTREND"
    except Exception:
        pass

    logger.warning(f"[CHAIN] Returning fallback NO_TRADE signal for {symbol}|{mode}")

    return SignalOutput(
        symbol               = symbol,
        mode                 = mode,
        signal               = SignalType.NO_TRADE,
        confidence           = 0.0,
        risk_level           = RiskLevel.HIGH,
        headline             = "Analysis unavailable — do not trade",
        summary              = (
            "The signal engine encountered an error during analysis. "
            "No actionable signal could be generated. "
            "Please retry or check data availability."
        ),
        caution              = "System error — treat as NO_TRADE until resolved",
        trade_setup          = None,
        timeframe_signals    = [
            {
                "timeframe": tf,
                "bias":      "neutral",
                "strength":  "weak",
                "key_note":  "Unavailable — analysis failed",
            }
            for tf in report.get("timeframes", {}).keys()
        ],
        confluence_factors   = [
            {
                "factor":    label,
                "alignment": "neutral",
                "detail":    "Unavailable — analysis failed",
            }
            for label in ("Trend", "RSI", "MACD", "BB", "SR")
        ],
        fake_signal_warning  = True,
        dominant_trend       = dominant_trend,
        bb_context           = "Unavailable — analysis failed",
    )


# ---------------------------------------------------------------------------
# Singleton — import this in task.py
# ---------------------------------------------------------------------------

_chain_instance: Optional[SignalChain] = None


def get_signal_chain() -> SignalChain:
    """
    Module-level singleton so the Gemini client is not re-initialised
    on every Celery task invocation.
    """
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = SignalChain()
    return _chain_instance
# src/agents/tools.py

"""
LangChain Tools
---------------
Each tool wraps a specific piece of the analysis pipeline so the agent
can call them selectively rather than receiving one giant JSON blob.

Tools available:
  1. GetSRZonesTool          — fetch support/resistance zones for a symbol+TF
  2. GetOrderBlocksTool      — fetch order blocks for a symbol+TF
  3. ComputeScoreTool        — compute composite score from indicator features
  4. CheckFakeSignalTool     — run fake signal detection on tf_features
  5. GetBBContextTool        — get Bollinger Band context for a symbol+TF
  6. GetGlobalConfluenceTool — compute multi-TF confluence from tf_data
  7. GetVolatilityRegimeTool — assess ATR-based volatility regime across TFs
"""

from __future__ import annotations

import json
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.helpers.reasoning import (
    compute_confluence,
    compute_score,
    detect_fake_signal,
    get_sr_zones,
    detect_order_blocks,
    detect_bb_context,
)
from src.helpers.indicators import add_indicators
from src.helpers.data_loader import process_symbol_multi_timeframe
from src.constants import MODE_TIMEFRAMES
from src.core.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------

class SRZonesInput(BaseModel):
    symbol:    str = Field(description="Trading pair e.g. BTCUSDT")
    timeframe: str = Field(description="Timeframe e.g. 1h, 4h, 1d")
    top_n:     int = Field(default=3, description="Number of zones per side")


class OrderBlocksInput(BaseModel):
    symbol:    str   = Field(description="Trading pair e.g. BTCUSDT")
    timeframe: str   = Field(description="Timeframe e.g. 1h, 4h, 1d")
    threshold: float = Field(default=1.5, description="Impulse threshold multiplier")
    top_n:     int   = Field(default=3, description="Max blocks per type")


class ComputeScoreInput(BaseModel):
    tf_features_json: str = Field(
        description="JSON string of a single timeframe's features dict containing "
                    "trend, rsi, macd, atr, bb keys"
    )


class FakeSignalInput(BaseModel):
    tf_features_json: str = Field(
        description="JSON string of a single timeframe's features dict"
    )


class BBContextInput(BaseModel):
    symbol:    str = Field(description="Trading pair e.g. BTCUSDT")
    timeframe: str = Field(description="Timeframe e.g. 1h, 4h, 1d")


class GlobalConfluenceInput(BaseModel):
    mode:         str = Field(description="SCALPER | SWING | POSITION")
    tf_data_json: str = Field(
        description="JSON string of {timeframe: {score: float, ...}} dict"
    )


class VolatilityRegimeInput(BaseModel):
    tf_data_json: str = Field(
        description="JSON string of tf_data dict containing atr features per timeframe"
    )


# ---------------------------------------------------------------------------
# Helper — load df for a single timeframe
# ---------------------------------------------------------------------------

def _load_df(symbol: str, timeframe: str):
    """Load and indicator-enrich a DataFrame for one symbol+timeframe."""
    results = process_symbol_multi_timeframe(
        symbol=symbol, timeframes=[timeframe]
    )
    if not results:
        raise ValueError(f"No data returned for {symbol} | {timeframe}")

    result = results[0]
    if result["status"] != "success" or result.get("rows", 0) == 0:
        raise ValueError(f"Data load failed for {symbol} | {timeframe}: {result}")

    return add_indicators(result["df"])


# ---------------------------------------------------------------------------
# Tool 1 — SR Zones
# ---------------------------------------------------------------------------

class GetSRZonesTool(BaseTool):
    name:        str = "get_sr_zones"
    description: str = (
        "Fetch the top support and resistance zones for a given symbol and timeframe. "
        "Returns a JSON dict with 'resistance' and 'support' lists, each containing "
        "{'level': float, 'strength': int} entries sorted by descending strength."
    )
    args_schema: Type[BaseModel] = SRZonesInput

    def _run(self, symbol: str, timeframe: str, top_n: int = 3) -> str:
        try:
            df     = _load_df(symbol, timeframe)
            zones  = get_sr_zones(df, top_n=top_n)
            logger.debug(f"[TOOL:SR_ZONES] {symbol}|{timeframe} → {zones}")
            return json.dumps(zones)
        except Exception as exc:
            logger.error(f"[TOOL:SR_ZONES ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Async not supported — use sync Celery workers")


# ---------------------------------------------------------------------------
# Tool 2 — Order Blocks
# ---------------------------------------------------------------------------

class GetOrderBlocksTool(BaseTool):
    name:        str = "get_order_blocks"
    description: str = (
        "Detect bullish and bearish order blocks for a given symbol and timeframe. "
        "Returns a JSON dict with 'bullish' and 'bearish' lists, each containing "
        "{'type', 'high', 'low'} dicts for the most recent qualifying blocks."
    )
    args_schema: Type[BaseModel] = OrderBlocksInput

    def _run(
        self,
        symbol:    str,
        timeframe: str,
        threshold: float = 1.5,
        top_n:     int   = 3,
    ) -> str:
        try:
            df     = _load_df(symbol, timeframe)
            blocks = detect_order_blocks(df, threshold=threshold, top_n=top_n)
            logger.debug(f"[TOOL:ORDER_BLOCKS] {symbol}|{timeframe}")
            return json.dumps(blocks)
        except Exception as exc:
            logger.error(f"[TOOL:ORDER_BLOCKS ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 3 — Composite Score
# ---------------------------------------------------------------------------

class ComputeScoreTool(BaseTool):
    name:        str = "compute_score"
    description: str = (
        "Compute the composite directional score [-1.0, 1.0] for a single timeframe. "
        "Pass the full tf_features dict (trend, rsi, macd, atr, bb) as a JSON string. "
        "Returns a JSON dict with key 'score'."
    )
    args_schema: Type[BaseModel] = ComputeScoreInput

    def _run(self, tf_features_json: str) -> str:
        try:
            tf = json.loads(tf_features_json)
            score = compute_score(
                trend=tf["trend"],
                rsi=tf["rsi"],
                macd=tf["macd"],
                atr=tf["atr"],
                bb=tf["bb"],
            )
            logger.debug(f"[TOOL:SCORE] score={score}")
            return json.dumps({"score": score})
        except Exception as exc:
            logger.error(f"[TOOL:SCORE ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 4 — Fake Signal Check
# ---------------------------------------------------------------------------

class CheckFakeSignalTool(BaseTool):
    name:        str = "check_fake_signal"
    description: str = (
        "Check whether a signal is likely fake based on indicator conflicts. "
        "Pass the full tf_features dict as a JSON string. "
        "Returns {'is_fake': bool, 'conflicts': list, 'low_volatility': bool}."
    )
    args_schema: Type[BaseModel] = FakeSignalInput

    def _run(self, tf_features_json: str) -> str:
        try:
            tf     = json.loads(tf_features_json)
            result = detect_fake_signal(tf)
            logger.debug(f"[TOOL:FAKE_SIGNAL] is_fake={result['is_fake']}")
            return json.dumps(result)
        except Exception as exc:
            logger.error(f"[TOOL:FAKE_SIGNAL ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 5 — BB Context
# ---------------------------------------------------------------------------

class GetBBContextTool(BaseTool):
    name:        str = "get_bb_context"
    description: str = (
        "Get Bollinger Band context for the latest candle on a given symbol+timeframe. "
        "Returns {'band_touch', 'inside_bands', 'squeeze', 'percent_b'}."
    )
    args_schema: Type[BaseModel] = BBContextInput

    def _run(self, symbol: str, timeframe: str) -> str:
        try:
            df     = _load_df(symbol, timeframe)
            result = detect_bb_context(df)
            logger.debug(f"[TOOL:BB_CONTEXT] {symbol}|{timeframe} → {result}")
            return json.dumps(result)
        except Exception as exc:
            logger.error(f"[TOOL:BB_CONTEXT ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 6 — Global Confluence
# ---------------------------------------------------------------------------

class GetGlobalConfluenceTool(BaseTool):
    name:        str = "get_global_confluence"
    description: str = (
        "Compute the weighted multi-timeframe confluence score for a given mode. "
        "Pass mode as a string and tf_data as a JSON string of "
        "{timeframe: {score: float, ...}}. "
        "Returns {'score', 'alignment', 'bullish_tfs', 'bearish_tfs'}."
    )
    args_schema: Type[BaseModel] = GlobalConfluenceInput

    def _run(self, mode: str, tf_data_json: str) -> str:
        try:
            tf_data = json.loads(tf_data_json)
            result  = compute_confluence(mode, tf_data)
            logger.debug(f"[TOOL:CONFLUENCE] mode={mode} → {result}")
            return json.dumps(result)
        except Exception as exc:
            logger.error(f"[TOOL:CONFLUENCE ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 7 — Volatility Regime
# ---------------------------------------------------------------------------

class GetVolatilityRegimeTool(BaseTool):
    name:        str = "get_volatility_regime"
    description: str = (
        "Assess the overall volatility regime across all timeframes. "
        "Pass tf_data JSON string — each TF must have an 'atr' key with "
        "'state' and 'volatility' fields. "
        "Returns {'regime': 'low'|'moderate'|'high', 'expanding_tfs': int, "
        "'contracting_tfs': int, 'detail': str}."
    )
    args_schema: Type[BaseModel] = VolatilityRegimeInput

    def _run(self, tf_data_json: str) -> str:
        try:
            tf_data = json.loads(tf_data_json)

            expanding   = 0
            contracting = 0
            high_vol    = 0
            low_vol     = 0

            for tf, data in tf_data.items():
                atr = data.get("atr", {})
                if atr.get("state") == "expanding":
                    expanding += 1
                else:
                    contracting += 1
                if atr.get("volatility") == "high":
                    high_vol += 1
                elif atr.get("volatility") == "low":
                    low_vol += 1

            total = len(tf_data) or 1

            if high_vol / total >= 0.5:
                regime = "high"
            elif low_vol / total >= 0.5:
                regime = "low"
            else:
                regime = "moderate"

            result = {
                "regime":          regime,
                "expanding_tfs":   expanding,
                "contracting_tfs": contracting,
                "detail": (
                    f"{expanding}/{total} TFs expanding ATR, "
                    f"{contracting}/{total} contracting"
                ),
            }

            logger.debug(f"[TOOL:VOLATILITY] {result}")
            return json.dumps(result)

        except Exception as exc:
            logger.error(f"[TOOL:VOLATILITY ERROR] {exc}", exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool Registry — import this in chain.py
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    GetSRZonesTool(),
    GetOrderBlocksTool(),
    ComputeScoreTool(),
    CheckFakeSignalTool(),
    GetBBContextTool(),
    GetGlobalConfluenceTool(),
    GetVolatilityRegimeTool(),
]
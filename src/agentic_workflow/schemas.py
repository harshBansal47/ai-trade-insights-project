# src/agents/schemas.py

"""
Central schema definitions for the signal agent.
All Pydantic models live here — import from this file everywhere.
"""

from __future__ import annotations
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    NEUTRAL     = "NEUTRAL"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_TRADE    = "NO_TRADE"


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"


# ---------------------------------------------------------------------------
# Nested Models
# ---------------------------------------------------------------------------

class EntryZone(BaseModel):
    low:  float = Field(description="Lower bound of suggested entry price zone")
    high: float = Field(description="Upper bound of suggested entry price zone")


class TradeSetup(BaseModel):
    entry_zone:    EntryZone      = Field(description="Recommended entry price zone")
    stop_loss:     float          = Field(description="Suggested stop-loss price level")
    take_profit_1: float          = Field(description="First take-profit target")
    take_profit_2: Optional[float] = Field(None, description="Second take-profit target")
    take_profit_3: Optional[float] = Field(None, description="Third take-profit target")
    risk_reward:   float          = Field(description="Risk/reward ratio for this setup")


class TimeframeSignal(BaseModel):
    timeframe: str = Field(description="e.g. 1h, 4h, 1d")
    bias:      str = Field(description="bullish | bearish | neutral")
    strength:  str = Field(description="strong | moderate | weak")
    key_note:  str = Field(description="One-line summary for this timeframe")


class ConfluenceFactor(BaseModel):
    factor:    str = Field(description="Indicator or pattern name")
    alignment: str = Field(description="bullish | bearish | neutral | warning")
    detail:    str = Field(description="Brief explanation of this factor's contribution")


# ---------------------------------------------------------------------------
# Root Output Model
# ---------------------------------------------------------------------------

class SignalOutput(BaseModel):
    """Complete structured signal output for the frontend."""

    # Core signal
    symbol:     str        = Field(description="Trading pair e.g. BTCUSDT")
    mode:       str        = Field(description="SCALPER | SWING | POSITION")
    signal:     SignalType = Field(description="Primary trading signal")
    confidence: float      = Field(description="Confidence score 0.0–1.0")
    risk_level: RiskLevel  = Field(description="Overall risk assessment")

    # Summary
    headline: str           = Field(description="One punchy headline sentence (max 15 words)")
    summary:  str           = Field(description="2–3 sentence analysis summary")
    caution:  Optional[str] = Field(None, description="Key risk or warning if any")

    # Setup
    trade_setup: Optional[TradeSetup] = Field(
        None,
        description="Concrete trade setup — null when signal is NO_TRADE or NEUTRAL",
    )

    # Supporting evidence
    timeframe_signals:  list[TimeframeSignal]  = Field(description="Per-timeframe bias breakdown")
    confluence_factors: list[ConfluenceFactor] = Field(description="Key contributing factors")

    # Meta
    fake_signal_warning: bool = Field(description="True if fake signal detected on any timeframe")
    dominant_trend:      str  = Field(description="UPTREND | DOWNTREND | SIDEWAYS")
    bb_context:          str  = Field(description="Brief Bollinger Band context note")
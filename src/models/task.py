import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON
from sqlalchemy import Enum as SAEnum


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class TaskStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class TradingMode(str, Enum):
    SCALPER      = "SCALPER"
    SWING        = "SWING"
    CONSERVATIVE = "CONSERVATIVE"


# ── DB Table ──────────────────────────────────────────────────────────────────

class Task(SQLModel, table=True):
    __tablename__ = "tasks"

    id: str = Field(default_factory=_uuid, primary_key=True, max_length=36)
    user_id: str = Field(foreign_key="users.id", index=True, max_length=36)
    celery_task_id: Optional[str] = Field(default=None, max_length=255)

    # Input
    coin: str = Field(max_length=100)
    coin_symbol: str = Field(max_length=20)
    mode: TradingMode = Field(
        default=TradingMode.SWING,
        sa_column=Column(SAEnum(TradingMode, name="trading_mode_enum"))
    )
    message: Optional[str] = Field(default=None, max_length=500)

    # State
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        sa_column=Column(SAEnum(TaskStatus, name="task_status_enum"))
    )
    error: Optional[str] = Field(default=None)

    # AI result stored as JSON blob
    result: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )

    # Points
    points_deducted: int = Field(default=0)

    # Timestamps
    created_at: datetime = Field(default_factory=_now, index=True)
    completed_at: Optional[datetime] = Field(default=None)

    def __repr__(self) -> str:
        return f"<Task id={self.id} coin={self.coin} status={self.status}>"


# ── API Schemas ───────────────────────────────────────────────────────────────

class AnalyzeRequest(SQLModel):
    coin: str = Field(min_length=1, max_length=100)
    coin_symbol: str = Field(min_length=1, max_length=20)
    mode: TradingMode
    message: Optional[str] = Field(default=None, max_length=500)


class AnalyzeResponse(SQLModel):
    task_id: str
    status: TaskStatus
    message: str = "Analysis started"


class Signal(SQLModel):
    type: str         # "buy" | "sell" | "hold"
    strength: str     # "weak" | "moderate" | "strong"
    description: str


class TechnicalIndicator(SQLModel):
    name: str
    value: str
    signal: str       # "bullish" | "bearish" | "neutral"


class AIInsight(SQLModel):
    action: str                    # "LONG" | "SHORT" | "WAIT"
    confidence: int                # 0-100
    trend: str                     # "UPTREND" | "DOWNTREND" | "SIDEWAYS"
    sentiment: str = "neutral"
    summary: str
    key_factors: list[str] = []
    risk_factors: list[str] = []
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: Optional[str] = None
    risk_level: str = "medium"     # "low" | "medium" | "high"
    recommendation: str
    market_analysis: str
    signals: list[Signal] = []
    technical_indicators: list[TechnicalIndicator] = []


class TaskStatusResponse(SQLModel):
    task_id: str
    status: TaskStatus
    coin: str
    coin_symbol: str
    mode: TradingMode
    message: Optional[str] = None
    result: Optional[AIInsight] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    points_deducted: int = 0

    model_config = {"from_attributes": True}


class HistoryItem(SQLModel):
    task_id: str
    coin: str
    coin_symbol: str
    mode: TradingMode
    status: TaskStatus
    result: Optional[AIInsight] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class HistoryResponse(SQLModel):
    items: list[HistoryItem]
    total: int
    page: int
    per_page: int
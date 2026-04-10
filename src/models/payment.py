import uuid
from datetime import datetime, timezone
from typing import Optional
from enum import Enum
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Enum as SAEnum


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class PaymentStatus(str, Enum):
    PENDING   = "pending"
    COMPLETED = "completed"
    FAILED    = "failed"
    REFUNDED  = "refunded"


class Payment(SQLModel, table=True):
    __tablename__ = "payments"

    id: str = Field(default_factory=_uuid, primary_key=True, max_length=36)
    user_id: str = Field(foreign_key="users.id", index=True, max_length=36)

    # Stripe
    stripe_session_id: Optional[str] = Field(
        default=None, unique=True, index=True, max_length=255
    )
    stripe_payment_intent_id: Optional[str] = Field(default=None, max_length=255)

    # Bundle
    bundle_id: str = Field(max_length=50)
    points: int
    amount_cents: int   # e.g. 500 = $5.00

    status: PaymentStatus = Field(
        default=PaymentStatus.PENDING,
        sa_column=Column(SAEnum(PaymentStatus, name="payment_status_enum"))
    )

    created_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = Field(default=None)

    def __repr__(self) -> str:
        return f"<Payment id={self.id} points={self.points} status={self.status}>"


# ── API Schemas ───────────────────────────────────────────────────────────────

POINTS_BUNDLES: list[dict] = [
    {"id": "starter", "name": "Starter", "points": 50,  "price_cents": 500,  "popular": False},
    {"id": "pro",     "name": "Pro",     "points": 150, "price_cents": 1200, "popular": True},
    {"id": "elite",   "name": "Elite",   "points": 500, "price_cents": 3500, "popular": False},
]


class CreateSessionRequest(SQLModel):
    bundle_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class CreateSessionResponse(SQLModel):
    session_id: str
    checkout_url: str


class VerifySessionResponse(SQLModel):
    success: bool
    points_added: int
    new_balance: int


class PaymentHistoryItem(SQLModel):
    id: str
    points: int
    amount_cents: int
    status: PaymentStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class PaymentHistoryResponse(SQLModel):
    items: list[PaymentHistoryItem]
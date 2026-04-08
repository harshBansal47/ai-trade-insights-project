import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Base fields shared across models ──────────────────────────────────────────

class UserBase(SQLModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(index=True, max_length=255)

    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    points: int = Field(default=0, ge=0)

    # Password (nullable for OAuth users)
    hashed_password: Optional[str] = Field(default=None, max_length=255)

    # ── OAuth fields ──
    provider: Optional[str] = Field(default=None, max_length=50)  
    # e.g. "google", "credentials"
    provider_id: Optional[str] = Field(default=None, index=True)  
    # e.g. Google "sub"
    image: Optional[str] = Field(default=None, max_length=500)


# ── Database table ────────────────────────────────────────────────────────────

class User(UserBase, table=True):
    __tablename__ = "users"

    id: str = Field(
        default_factory=_uuid,
        primary_key=True,
        max_length=36,
    )

    # Enforce unique email at DB level
    __table_args__ = (
        {"sqlite_autoincrement": True},  # optional (safe default)
    )

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} provider={self.provider}>"
    

# ── API schemas (Pydantic only) ───────────────────────────────────────────────

class UserCreate(SQLModel):
    name: str = Field(min_length=2, max_length=120)
    email: str
    password: str = Field(min_length=8, max_length=128)
    otp: str = Field(min_length=6, max_length=6)


class OAuthUserCreate(SQLModel):
    """For Google / OAuth signup"""
    name: str
    email: str
    provider: str  # "google"
    provider_id: str  # Google sub
    image: Optional[str] = None


class UserRead(SQLModel):
    """Shape returned in auth responses"""
    id: str
    name: str
    email: str
    image: Optional[str]
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(SQLModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    image: Optional[str] = None
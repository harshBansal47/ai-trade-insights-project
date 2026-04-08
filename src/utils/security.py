from datetime import datetime, timedelta, timezone
from typing import Any
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import HTTPException, status
from src.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

jwt_algorithm: str = "HS256"
access_token_expire_minutes: int = 10080

# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=access_token_expire_minutes)
    )
    payload.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(payload, settings.secret_key, algorithm=jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[jwt_algorithm],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def extract_user_id(token: str) -> str:
    payload = decode_token(token)
    uid: str | None = payload.get("sub")
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )
    return uid
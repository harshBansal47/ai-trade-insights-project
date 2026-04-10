import random
import string
from fastapi import status
from fastapi import HTTPException
from src.core.redis import get_redis_client
from src.core.config import settings



def _key(email: str, purpose: str) -> str:
    return f"otp:{purpose}:{email.lower().strip()}"


def _generate() -> str:
    return "".join(random.choices(string.digits, k=settings.otp_length))


async def create_otp(email: str, purpose: str) -> str:
    """Generate a fresh OTP, store in Redis with TTL, and return the code."""
    code = _generate()
    redis = await get_redis_client()
    await redis.setex(_key(email, purpose), settings.otp_expire_minutes * 60, code)
    return code


async def verify_otp(email: str, code: str, purpose: str) -> bool:
    """Return True and delete the stored code if it matches; False otherwise."""
    redis = await get_redis_client()
    stored = await redis.get(_key(email, purpose))
    normalized_stored = stored.decode('utf-8') if stored else None
    if stored and normalized_stored == code:
        await redis.delete(_key(email, purpose))
        return True
    return False


async def delete_otp(email: str, purpose: str) -> None:
    redis = await get_redis_client()
    await redis.delete(_key(email, purpose))


# ── shared OTP verification helper ───────────────────────────────────────────

async def _require_otp_verified(email: str, purpose: str) -> None:
    """
    Verifies the short-lived flag stored by /verify-otp.
    Raises HTTP 400 if missing or expired.
    """
    redis = await get_redis_client()
    flag_key = f"otp_verified:{purpose}:{email}"
    flag = await redis.get(flag_key)
    if not flag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code",
        )
    await redis.delete(flag_key)
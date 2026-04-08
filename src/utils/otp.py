import random
import string
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
    if stored and stored == code:
        await redis.delete(_key(email, purpose))
        return True
    return False


async def delete_otp(email: str, purpose: str) -> None:
    redis = await get_redis_client()
    await redis.delete(_key(email, purpose))
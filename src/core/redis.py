from redis import ConnectionPool, Redis
from redis.asyncio import ConnectionPool as AsyncConnectionPool, Redis as AsyncRedis

REDIS_URL = "redis://localhost:6379"

# ── Sync ──────────────────────────────────────────────────────────────────
_sync_pool = ConnectionPool.from_url(
    REDIS_URL,
    max_connections=30,
    decode_responses=True,
)
_sync_client = Redis(connection_pool=_sync_pool)


# ── Async ─────────────────────────────────────────────────────────────────
_async_pool = AsyncConnectionPool.from_url(
    REDIS_URL,
    max_connections=30,
    decode_responses=True,
)
_async_client = AsyncRedis(connection_pool=_async_pool)


# ── Public API ────────────────────────────────────────────────────────────
def get_redis_client() -> Redis:
    return _sync_client


def get_async_redis_client() -> AsyncRedis:
    return _async_client         
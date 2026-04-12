from redis import ConnectionPool, Redis

REDIS_URL = "redis://localhost:6379"

pool = ConnectionPool.from_url(
    REDIS_URL,
    max_connections=30,
    decode_responses=True,
)

redis_client = Redis(connection_pool=pool)


def get_redis_client() -> Redis:
    return redis_client
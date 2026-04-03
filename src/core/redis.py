import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from typing import Optional, AsyncIterator

REDIS_URL = "redis://localhost:6379"

class RedisClient:
    _instance: Optional["RedisClient"] = None
    _client: Optional[redis.Redis] = None
    _pool: Optional[ConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        try:
            self._pool = ConnectionPool.from_url(
                REDIS_URL,
                max_connections=30,
                decode_responses=False,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            await self._client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print(f"❌ Redis init failed: {e}")
            raise

    async def connect(self) -> redis.Redis:
        if not self._client:
            await self.initialize()
        return self._client


    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        print("🛑 Redis closed")




async def get_redis_client() -> AsyncIterator[redis.Redis]:
    app_redis_client = RedisClient()
    client = await app_redis_client.connect()
    return client
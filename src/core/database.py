from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
from src.core.config import settings


class DatabaseManager:
    _instance: Optional["DatabaseManager"] = None

    def __init__(self):
        if DatabaseManager._instance is not None:
            raise Exception("Use get_instance() instead")

        # ✅ Engine (Supabase-safe config)
        self.engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            poolclass=NullPool,  # ✅ IMPORTANT for Supabase
            connect_args={"statement_cache_size": 0},
        )

        # ✅ Session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        DatabaseManager._instance = self

    # ── Singleton Access ─────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance

    # ── Startup Hook ─────────────────────────────────────────────
    async def connect(self):
        """Called on FastAPI startup"""
        # Test DB connection
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda conn: None)

        print("✅ Database & Redis connected")

    # ── Shutdown Hook ─────────────────────────────────────────────
    async def disconnect(self):
        await self.engine.dispose()
        print("🔴 Database & Redis disconnected")

    # ── Session Generator ─────────────────────────────────────────
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ── Manual Session (if needed outside DI) ─────────────────────
    async def session(self) -> AsyncSession:
        return self.session_factory()
    

# ═══════════════════════════════════════════════════════
# DATABASE DEPENDENCY
# Wraps DatabaseManager.get_session() (async generator)
# into a FastAPI-compatible Depends()-able function.
# All endpoints receive `db` via Depends(get_db).
# ═══════════════════════════════════════════════════════

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a session from the singleton
    DatabaseManager. The session is committed on success and
    rolled back on any exception — exactly as the manager defines.
    """
    async for session in DatabaseManager.get_instance().get_session():
        yield session
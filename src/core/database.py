from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
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

        # ── ASYNC engine — FastAPI routes ────────────────────────────────
        self._async_engine = create_async_engine(
            settings.database_url_async,    
            echo=settings.debug,
            pool_pre_ping=True,
            poolclass=NullPool,
            connect_args={"statement_cache_size": 0},
        )

        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # ── SYNC engine — Celery tasks ───────────────────────────────────
        self._sync_engine = create_engine(
            settings.database_url_sync,        
            echo=settings.debug,
            pool_pre_ping=True,
            poolclass=NullPool,
        )

        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            class_=Session,
            expire_on_commit=False,
        )

        DatabaseManager._instance = self

    # ── Singleton ────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance

    # ── Lifecycle ────────────────────────────────────────────────────────
    async def connect(self):
        """Called on FastAPI startup — tests async connection."""
        async with self._async_engine.begin() as conn:
            await conn.run_sync(lambda conn: None)
        print("✅ Database connected")

    async def disconnect(self):
        """Called on FastAPI shutdown — disposes both engines."""
        await self._async_engine.dispose()
        self._sync_engine.dispose()
        print("🔴 Database disconnected")

    # ── ASYNC session — for FastAPI / async code ─────────────────────────
    async def open_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Yields a managed AsyncSession.
        Commits on success, rolls back on exception, always closes.

        Used by: get_async_db() FastAPI dependency.
        """
        async with self._async_session_factory() as async_session:
            try:
                yield async_session
                await async_session.commit()
            except Exception:
                await async_session.rollback()
                raise

    # ── SYNC session — for Celery tasks ──────────────────────────────────
    def open_sync_session(self) -> Session:
        """
        Returns a sync Session as a context manager.
        Caller is responsible for commit/rollback.

        Used by: get_sync_db() Celery helper.

        Usage:
            with DatabaseManager.get_instance().open_sync_session() as sync_db:
                task = sync_db.get(Task, task_id)
                task.status = TaskStatus.COMPLETED
                sync_db.commit()
        """
        return self._sync_session_factory()


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Inject an AsyncSession into FastAPI route handlers.

        async def my_route(async_db: AsyncSession = Depends(get_async_db)):
            ...
    """
    async for async_session in DatabaseManager.get_instance().open_async_session():
        yield async_session


# ── Celery helper ─────────────────────────────────────────────────────────────

def get_sync_db() -> Session:
    """
    Returns a sync Session context manager for use inside Celery tasks.

        with get_sync_db() as sync_db:
            task = sync_db.get(Task, task_id)
            ...
    """
    return DatabaseManager.get_instance().open_sync_session()
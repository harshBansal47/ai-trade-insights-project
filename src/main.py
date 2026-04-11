from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel
from src.models import *
from src.core.config import settings
from src.core.database import DatabaseManager
from src.routes import user as auth
from src.routes import tasks as task

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_manager = DatabaseManager.get_instance()
    await db_manager.connect()

    async with db_manager.engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        
    yield
    await db_manager.disconnect()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="AI-powered crypto trading analysis API",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router)
    app.include_router(task.router)


    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["health"], include_in_schema=False)
    async def health():
        return JSONResponse({"status": "ok", "app": settings.app_name})

    return app


app = create_app()

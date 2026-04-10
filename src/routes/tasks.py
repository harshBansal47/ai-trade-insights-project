from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import select, func, desc
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import get_session
from app.middleware.auth import get_current_user
from app.models.user import User
from app.models.task import (
    Task, TaskStatus, TradingMode,
    AnalyzeRequest, AnalyzeResponse,
    TaskStatusResponse, HistoryItem, HistoryResponse, AIInsight,
)
from app.config import settings

router = APIRouter(tags=["tasks"])


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze(
    body: AnalyzeRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Points check
    if current_user.points < settings.points_per_query:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient points. Need {settings.points_per_query}, have {current_user.points}.",
        )

    # Deduct optimistically (refunded on failure inside the worker)
    current_user.points -= settings.points_per_query
    db.add(current_user)

    # Create task record
    task = Task(
        user_id=current_user.id,
        coin=body.coin,
        coin_symbol=body.coin_symbol.upper(),
        mode=body.mode,
        message=body.message,
        status=TaskStatus.PENDING,
        points_deducted=settings.points_per_query,
    )
    db.add(task)
    await db.flush()   # get task.id before commit

    # Dispatch Celery task
    from app.workers.tasks import run_analysis_task
    celery_job = run_analysis_task.delay(
        task.id,
        current_user.id,
        task.coin,
        task.coin_symbol,
        task.mode.value,
        task.message,
    )
    task.celery_task_id = celery_job.id
    db.add(task)

    return AnalyzeResponse(task_id=task.id, status=TaskStatus.PENDING)


# ── GET /task-status/{task_id} ────────────────────────────────────────────────

@router.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await db.exec(
        select(Task).where(Task.id == task_id, Task.user_id == current_user.id)
    )
    task = result.first()

    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    insight = AIInsight(**task.result) if task.result else None

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        coin=task.coin,
        coin_symbol=task.coin_symbol,
        mode=task.mode,
        message=task.message,
        result=insight,
        error=task.error,
        created_at=task.created_at,
        completed_at=task.completed_at,
        points_deducted=task.points_deducted,
    )


# ── GET /history ──────────────────────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    offset = (page - 1) * per_page

    # Total count
    count_res = await db.exec(
        select(func.count()).where(Task.user_id == current_user.id).select_from(Task)
    )
    total = count_res.one()

    # Page of tasks
    tasks_res = await db.exec(
        select(Task)
        .where(Task.user_id == current_user.id)
        .order_by(desc(Task.created_at))
        .offset(offset)
        .limit(per_page)
    )
    tasks = tasks_res.all()

    items = [
        HistoryItem(
            task_id=t.id,
            coin=t.coin,
            coin_symbol=t.coin_symbol,
            mode=t.mode,
            status=t.status,
            result=AIInsight(**t.result) if t.result else None,
            created_at=t.created_at,
            completed_at=t.completed_at,
        )
        for t in tasks
    ]

    return HistoryResponse(items=items, total=total, page=page, per_page=per_page)


# ── GET /history/{task_id} ────────────────────────────────────────────────────

@router.get("/history/{task_id}", response_model=TaskStatusResponse)
async def get_history_item(
    task_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await db.exec(
        select(Task).where(Task.id == task_id, Task.user_id == current_user.id)
    )
    task = result.first()

    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    insight = AIInsight(**task.result) if task.result else None

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        coin=task.coin,
        coin_symbol=task.coin_symbol,
        mode=task.mode,
        message=task.message,
        result=insight,
        error=task.error,
        created_at=task.created_at,
        completed_at=task.completed_at,
        points_deducted=task.points_deducted,
    )
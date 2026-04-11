from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func, desc
from src.services.analysis import run_analysis
from src.core.database import get_db
from src.middlewares.auth import get_current_user
from src.models.task import AIInsight, AnalyzeRequest, AnalyzeResponse, HistoryItem, HistoryResponse, Task, TaskStatus, TaskStatusResponse
from src.models.user import User
from src.core.config import settings

router = APIRouter(tags=["tasks"])


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze(
    body: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.points < settings.points_per_query:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient points. Need {settings.points_per_query}, have {current_user.points}.",
        )

    current_user.points -= settings.points_per_query
    db.add(current_user)

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
    await db.flush()

    celery_job = run_analysis.delay(
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Task).where(Task.id == task_id, Task.user_id == current_user.id)
    )
    task = result.scalar_one_or_none()

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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    offset = (page - 1) * per_page

    count_res = await db.execute(
        select(func.count()).where(Task.user_id == current_user.id).select_from(Task)
    )
    total = count_res.scalar_one()

    tasks_res = await db.execute(
        select(Task)
        .where(Task.user_id == current_user.id)
        .order_by(desc(Task.created_at))
        .offset(offset)
        .limit(per_page)
    )
    tasks = tasks_res.scalars().all()

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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Task).where(Task.id == task_id, Task.user_id == current_user.id)
    )
    task = result.scalar_one_or_none()

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
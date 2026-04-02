from celery import Celery
from celery.result import AsyncResult


celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1", 
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600, 
)


def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task.status,
    }

    if task.status == "SUCCESS":
        response["result"] = task.result

    elif task.status == "FAILURE":
        response["error"] = str(task.result)

    return response
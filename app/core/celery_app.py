from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.imports = ("app.services.ingestion.tasks",)

celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1
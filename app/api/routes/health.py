import time
from collections.abc import Callable
from typing import Any

import boto3
import chromadb
import redis
from botocore.config import Config
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.core.warmup import models_warmed
from app.db.session import SessionLocal
from app.services.ollama_client import ollama_client

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

settings = get_settings()


def _probe_postgres() -> None:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))


def _probe_redis() -> None:
    client = redis.Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    client.ping()


def _probe_chroma() -> None:
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    client.heartbeat()


def _probe_s3() -> None:
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=Config(
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 1},
        ),
    )
    head_bucket_kwargs = {"Bucket": settings.s3_bucket_name}
    if settings.s3_expected_bucket_owner:
        head_bucket_kwargs["ExpectedBucketOwner"] = settings.s3_expected_bucket_owner
    client.head_bucket(**head_bucket_kwargs)


def _probe_ollama() -> None:
    ollama_client.list()


HEALTH_PROBES: dict[str, Callable[[], None]] = {
    "postgres": _probe_postgres,
    "redis": _probe_redis,
    "chroma": _probe_chroma,
    "s3": _probe_s3,
    "ollama": _probe_ollama,
}


def _run_check(probe: Callable[[], None]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        probe()
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc.__class__.__name__,
        }


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    checks = {name: _run_check(probe) for name, probe in HEALTH_PROBES.items()}
    model_status = not settings.warm_models_on_startup or models_warmed()
    ready_status = all(check["ok"] for check in checks.values()) and model_status

    if not ready_status:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if ready_status else "degraded",
        "checks": checks,
        "models_warmed": models_warmed(),
    }

import time
from typing import Any

import boto3
import chromadb
import redis
from botocore.config import Config
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.ollama_client import ollama_client

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

settings = get_settings()


def check_postgres() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc.__class__.__name__,
        }


def check_redis() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc.__class__.__name__,
        }


def check_chroma() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        client.heartbeat()
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc.__class__.__name__,
        }


def check_s3() -> dict[str, Any]:
    start = time.perf_counter()
    try:
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
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc.__class__.__name__,
        }


def check_ollama() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        ollama_client.list()
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
    checks = {
        "postgres": check_postgres(),
        "redis": check_redis(),
        "chroma": check_chroma(),
        "s3": check_s3(),
        "ollama": check_ollama(),
    }
    ready_status = all(check["ok"] for check in checks.values())

    if not ready_status:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if ready_status else "degraded",
        "checks": checks,
    }

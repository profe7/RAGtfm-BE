from contextlib import asynccontextmanager

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.concurrency import run_in_threadpool
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import auth
from app.api.routes.documents import router as documents_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.rag import router as rag_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.test_metrics import router as test_metrics_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import ObservabilityMiddleware
from app.core.rate_limit import limiter
from app.core.warmup import warm_models

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger("app.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", app=settings.app_name, version=settings.app_version)
    if settings.warm_models_on_startup:
        await run_in_threadpool(warm_models)
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(ObservabilityMiddleware)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(ingestion_router)
api_router.include_router(retrieval_router)
api_router.include_router(rag_router)
api_router.include_router(test_metrics_router)
api_router.include_router(events_router)
api_router.include_router(documents_router)
api_router.include_router(health_router)
api_router.include_router(auth.router)

app.include_router(api_router)
app.mount("/metrics", make_asgi_app())

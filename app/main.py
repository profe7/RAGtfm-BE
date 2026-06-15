from fastapi import FastAPI, APIRouter

from app.api.routes import auth
from app.api.routes.documents import router as documents_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.rag import router as rag_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.test_metrics import router as test_metrics_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

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

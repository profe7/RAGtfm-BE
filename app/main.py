from fastapi import FastAPI

from app.api.routes import auth
from app.api.routes.documents import router as documents_router
from app.api.routes.events import router as events_router
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

app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(rag_router)
app.include_router(test_metrics_router)
app.include_router(events_router) 
app.include_router(documents_router)
app.include_router(auth.router)

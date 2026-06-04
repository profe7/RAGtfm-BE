from fastapi import FastAPI

from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.rag import router as rag_router
from app.api.routes.test_metrics import router as test_metrics_router
from app.api.routes.documents import router as documents_router
from app.db.session import init_db
from contextlib import asynccontextmanager
from app.core.config import get_settings
from app.api.routes import auth

settings = get_settings()
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(rag_router)
app.include_router(test_metrics_router)
app.include_router(documents_router)
app.include_router(auth.router)

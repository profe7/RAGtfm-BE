from fastapi import FastAPI

from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.rag import router as rag_router
from app.api.routes.test_metrics import router as test_metrics_router


app = FastAPI(
    title="RAGtfm API",
    version="0.1.0",
)

app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(rag_router)
app.include_router(test_metrics_router)

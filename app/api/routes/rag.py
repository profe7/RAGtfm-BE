from time import perf_counter
from fastapi import APIRouter
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.generation.ollama_generator import generate_answer
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.utils.timing import timed_stage
from app.api.deps import get_current_user
from app.db.models import UserRecord
from fastapi import Depends
from typing import Annotated


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post("/query", response_model=RagQueryResponse)
def query_rag(request: RagQueryRequest, current_user: Annotated[UserRecord, Depends(get_current_user)]):
    metrics = {}
    total_start = perf_counter()

    with timed_stage(metrics, "retrieval_ms"):
        chunks = retrieve_hybrid_chunks(
            query=request.query,
            limit=request.limit,
            candidate_limit=20,
            metrics=metrics,
            user_id=current_user.id,
            document_ids=request.document_ids, 
        )

    with timed_stage(metrics, "generation_ms"):
        answer = generate_answer(
            query=request.query,
            chunks=chunks,
        )

    metrics["total_ms"] = round((perf_counter() - total_start) * 1000, 2)

    return {
        "query": request.query,
        "answer": answer,
        "metrics": metrics,
        "sources": chunks,
    }
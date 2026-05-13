from pydantic import BaseModel, Field
from fastapi import APIRouter
from time import perf_counter

from app.services.generation.ollama_generator import generate_answer
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.utils.timing import timed_stage



router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/query")
def query_rag(request: RagQueryRequest):
    metrics = {}
    total_start = perf_counter()

    with timed_stage(metrics, "retrieval_ms"):
        chunks = retrieve_hybrid_chunks(
            query=request.query,
            limit=request.limit,
            candidate_limit=20,
            metrics=metrics,
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
        "sources": [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "distance": chunk.get("distance"),
                "rrf_score": chunk.get("rrf_score"),
                "retrieval_sources": chunk.get("retrieval_sources"),
                "dense_rank": chunk.get("dense_rank"),
                "bm25_rank": chunk.get("bm25_rank"),
                "rerank_score": chunk.get("rerank_score"),
                "rerank_rank": chunk.get("rerank_rank"),
                "metadata": chunk["metadata"],
            }
            for chunk in chunks
        ],
    }


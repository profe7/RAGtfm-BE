import json
from time import perf_counter
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.rag import RagQueryRequest
from app.services.generation.ollama_generator import generate_answer
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.services.retrieval.query_rewriter import rewrite_query_hyde
from app.utils.timing import timed_stage
from app.api.deps import get_current_user
from app.db.models import UserRecord
from fastapi import Depends
from typing import Annotated


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post("/query")
async def query_rag(request: RagQueryRequest, current_user: Annotated[UserRecord, Depends(get_current_user)]):
    metrics = {}
    total_start = perf_counter()

    with timed_stage(metrics, "query_rewrite_ms"):
        retrieval_query = rewrite_query_hyde(request.query)

    with timed_stage(metrics, "retrieval_ms"):
        chunks = retrieve_hybrid_chunks(
            query=request.query,
            dense_query=retrieval_query,
            limit=request.limit,
            candidate_limit=20,
            metrics=metrics,
            user_id=current_user.id,
            document_ids=request.document_ids,
        )
    
    async def response_generator():
        yield json.dumps({"type": "sources", "data": chunks}, separators=(',', ':')) + "\n"

        with timed_stage(metrics, "generation_ms"):
            async for token in generate_answer(query=request.query, chunks=chunks):
                yield json.dumps({"type": "token", "data": token}, separators=(',', ':')) + "\n"

        metrics["total_ms"] = round((perf_counter() - total_start) * 1000, 2)
        yield json.dumps({"type": "metrics", "data": metrics}, separators=(',', ':')) + "\n"

    return StreamingResponse(
        response_generator(),
        media_type="application/x-ndjson"
    )


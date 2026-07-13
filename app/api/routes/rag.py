import json
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.metrics import RAG_QUERIES, observe_rag_stages
from app.core.rate_limit import limiter, user_or_ip
from app.db.models import UserRecord
from app.db.session import get_db
from app.schemas.rag import RagQueryRequest
from app.services.conversations.conversation_store import (
    append_message,
    get_or_create_conversation,
    load_recent_messages,
)
from app.services.generation.ollama_generator import generate_answer
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.services.retrieval.query_rewriter import contextualize_query, rewrite_query_hyde
from app.services.retrieval.retrieval_filter import RetrievalFilter
from app.utils.timing import timed_stage

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)

settings = get_settings()


def _persist_turn(db: Session, conversation_id: str, user_query: str, answer: str) -> None:
    append_message(db, conversation_id, "user", user_query)
    if answer:
        append_message(db, conversation_id, "assistant", answer)


@router.post("/query")
@limiter.limit(settings.rag_query_rate_limit, key_func=user_or_ip)
async def query_rag(
    request: Request,
    payload: RagQueryRequest,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    RAG_QUERIES.inc()
    metrics = {}
    total_start = perf_counter()

    conversation_id = await run_in_threadpool(
        get_or_create_conversation, db, current_user.id, payload.conversation_id
    )
    history = await run_in_threadpool(
        load_recent_messages, db, conversation_id, settings.conversation_history_turns
    )

    # Offload the blocking, sync retrieval steps to a thread so they don't stall the
    # event loop of this async handler (see PRIVATENOTES §10).
    with timed_stage(metrics, "query_rewrite_ms"):
        standalone_query = await run_in_threadpool(contextualize_query, history, payload.query)
        retrieval_query = await run_in_threadpool(rewrite_query_hyde, standalone_query)

    with timed_stage(metrics, "retrieval_ms"):
        chunks = await run_in_threadpool(
            retrieve_hybrid_chunks,
            query=standalone_query,
            dense_query=retrieval_query,
            limit=payload.limit,
            candidate_limit=20,
            metrics=metrics,
            retrieval_filter=RetrievalFilter(
                user_id=current_user.id,
                document_ids=payload.document_ids,
            ),
        )

    async def response_generator():
        yield (
            json.dumps(
                {"type": "conversation", "data": {"conversation_id": conversation_id}},
                separators=(",", ":"),
            )
            + "\n"
        )
        yield json.dumps({"type": "sources", "data": chunks}, separators=(",", ":")) + "\n"

        answer_parts: list[str] = []
        try:
            with timed_stage(metrics, "generation_ms"):
                async for token in generate_answer(
                    query=payload.query, chunks=chunks, history=history
                ):
                    answer_parts.append(token)
                    yield (
                        json.dumps({"type": "token", "data": token}, separators=(",", ":")) + "\n"
                    )

            metrics["total_ms"] = round((perf_counter() - total_start) * 1000, 2)
            yield json.dumps({"type": "metrics", "data": metrics}, separators=(",", ":")) + "\n"
        finally:
            observe_rag_stages(metrics)
            await run_in_threadpool(
                _persist_turn, db, conversation_id, payload.query, "".join(answer_parts)
            )

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

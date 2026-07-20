from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.db.models import UserRecord
from app.schemas.retrieval import RetrieveChunksResponse
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.services.retrieval.retrieval_filter import RetrievalFilter
from app.services.retrieval.source_presenter import present_retrieved_chunks

router = APIRouter(
    prefix="/retrieve",
    tags=["Retrieval"],
)


@router.get("/chunks", response_model=RetrieveChunksResponse)
def retrieve_chunks(
    query: Annotated[str, Query(min_length=1)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    document_ids: Annotated[
        list[str] | None,
        Query(description="Optional document IDs owned by the current user"),
    ] = None,
):
    chunks = retrieve_hybrid_chunks(
        query=query,
        limit=limit,
        candidate_limit=10,
        retrieval_filter=RetrievalFilter(
            user_id=current_user.id,
            document_ids=document_ids,
        ),
    )

    return {
        "query": query,
        "count": len(chunks),
        "chunks": present_retrieved_chunks(chunks),
    }

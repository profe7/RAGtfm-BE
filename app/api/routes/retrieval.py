from typing import Annotated

from fastapi import APIRouter, Query
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks


router = APIRouter(
    prefix="/retrieve",
    tags=["Retrieval"],
)


@router.get("/chunks")
def retrieve_chunks(
    query: Annotated[str, Query(min_length=1)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
):
    chunks = retrieve_hybrid_chunks(
        query=query,
        limit=limit,
        candidate_limit=10,
    )

    return {
        "query": query,
        "count": len(chunks),
        "chunks": chunks,
    }

from typing import Any

from pydantic import BaseModel


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    rrf_score: float | None = None
    retrieval_sources: list[str] | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


class RetrieveChunksResponse(BaseModel):
    query: str
    count: int
    chunks: list[RetrievedChunkResponse]
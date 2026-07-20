from typing import Any

from pydantic import BaseModel


class SourceLocationResponse(BaseModel):
    element_id: str | None = None
    source_order: int | None = None
    element_type: str | None = None
    page_number: int | None = None
    coordinates: dict[str, Any] | None = None


class CitationSourceResponse(BaseModel):
    document_id: str | None = None
    filename: str | None = None
    chunk_type: str | None = None
    page_numbers: list[int]
    source_locations: list[SourceLocationResponse]


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    citation: CitationSourceResponse
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

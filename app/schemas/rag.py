from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievedChunkResponse


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class RagQueryResponse(BaseModel):
    query: str
    answer: str
    metrics: dict[str, float]
    sources: list[RetrievedChunkResponse]
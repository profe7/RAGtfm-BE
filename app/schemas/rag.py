from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Optional list of document IDs to restrict the search to"
    )

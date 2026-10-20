from pydantic import BaseModel, Field


class StoredDocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str


class IngestDocumentResponse(BaseModel):
    document_id: str
    document: StoredDocumentResponse
    filename: str | None
    status: str
    chunk_count: int
    stored_chunk_count: int
    stored_chunk_ids: list[str] = Field(default_factory=list)


class IngestPdfResponse(IngestDocumentResponse):
    pass

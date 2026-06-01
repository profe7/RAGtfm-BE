from pydantic import BaseModel


class StoredDocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    storage_backend: str
    storage_uri: str
    storage_path: str


class IngestPdfResponse(BaseModel):
    document_id: str
    document: StoredDocumentResponse
    filename: str | None
    chunk_count: int
    stored_chunk_count: int
    stored_chunk_ids: list[str]
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    storage_backend: str
    storage_uri: str
    storage_path: str
    status: str
    chunk_count: int
    stored_chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    documents: list[DocumentResponse]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_deleted: bool
    object_deleted: bool

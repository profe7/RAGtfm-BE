from typing import Annotated
from app.core.config import get_settings
from app.services.documents.document_catalog import delete_document_record
from app.services.documents.document_storage import delete_document_from_s3_storage
from app.services.vectorstores.chroma_store import delete_document_chunks
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.documents.document_catalog import (
    get_document_record,
    list_document_records,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

settings = get_settings()


def document_record_to_response(document) -> dict:
    return {
        "document_id": document.document_id,
        "original_filename": document.original_filename,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "sha256": document.sha256,
        "storage_backend": document.storage_backend,
        "storage_uri": document.storage_uri,
        "storage_path": document.storage_path,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "stored_chunk_count": document.stored_chunk_count,
        "created_at": document.created_at.isoformat(),
    }


@router.get("")
def list_documents(db: Annotated[Session, Depends(get_db)]):
    documents = list_document_records(db)

    return {
        "count": len(documents),
        "documents": [
            document_record_to_response(document)
            for document in documents
        ],
    }


@router.get(
    "/{document_id}",
    responses={404: {"description": "Document not found"}},
)
def get_document(document_id: Annotated[str, Path(min_length=1)], db: Annotated[Session, Depends(get_db)],):
    document = get_document_record(
        db=db,
        document_id=document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document_record_to_response(document)

@router.delete(
    "/{document_id}",
    responses={404: {"description": "Document not found"}},
)
def delete_document(
    document_id: Annotated[str, Path(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
):
    document = get_document_record(
        db=db,
        document_id=document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    delete_document_chunks(document_id=document_id)

    object_deleted = delete_document_from_s3_storage(
        storage_path=document.storage_path,
        endpoint_url=settings.s3_endpoint_url,
        access_key_id=settings.s3_access_key_id,
        secret_access_key=settings.s3_secret_access_key,
        bucket_name=settings.s3_bucket_name,
        region=settings.s3_region,
        s3_expected_bucket_owner=settings.s3_expected_bucket_owner,
    )

    delete_document_record(
        db=db,
        document_id=document_id,
    )

    return {
        "document_id": document_id,
        "deleted": True,
        "chunks_deleted": True,
        "object_deleted": object_deleted,
    }
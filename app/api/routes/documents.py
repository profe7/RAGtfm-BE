from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.documents import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.documents.document_catalog import (
    delete_document_record,
    get_document_record,
    list_document_records,
)
from app.services.documents.document_storage import delete_document_from_s3_storage
from app.services.vectorstores.chroma_store import delete_document_chunks


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

settings = get_settings()


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Annotated[Session, Depends(get_db)]):
    documents = list_document_records(db)

    return {
        "count": len(documents),
        "documents": documents,
    }


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"description": "Document not found"}},
)
def get_document(
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

    return document


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
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
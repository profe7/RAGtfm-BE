import math
from typing import Annotated
from urllib.parse import quote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.constants import DocumentStatus
from app.db.models import UserRecord
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
from app.services.documents.document_storage import (
    delete_document_from_s3_storage,
    download_document_from_s3_storage,
)
from app.services.retrieval.bm25_retriever import clear_bm25_cache
from app.services.vectorstores.chroma_store import delete_document_chunks

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

settings = get_settings()
logger = structlog.get_logger(__name__)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(5, ge=1, le=100, description="Items per page"),
):
    skip = (page - 1) * page_size
    documents, total = list_document_records(db, current_user.id, skip, page_size)
    pages = max(1, math.ceil(total / page_size))

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
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
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    document = get_document_record(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document


@router.get(
    "/{document_id}/content",
    response_class=Response,
    responses={
        200: {"content": {"application/pdf": {}}},
        404: {"description": "Document not found"},
        503: {"description": "Document content is temporarily unavailable"},
    },
)
async def get_document_content(
    document_id: Annotated[str, Path(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    document = get_document_record(db, document_id, current_user.id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        content = await run_in_threadpool(
            download_document_from_s3_storage,
            storage_path=document.storage_path,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            bucket_name=settings.s3_bucket_name,
            region=settings.s3_region,
            s3_expected_bucket_owner=settings.s3_expected_bucket_owner,
        )
    except Exception:
        logger.exception(
            "document_content_download_failed",
            document_id=document_id,
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=503,
            detail="Document content is temporarily unavailable",
        ) from None

    encoded_filename = quote(document.original_filename, safe="")
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={
        404: {"description": "Document not found"},
        409: {"description": "Document is being processed, unable to delete"},
    },
)
def delete_document(
    document_id: Annotated[str, Path(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    document = get_document_record(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a document that is currently being processed.",
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
        user_id=current_user.id,
    )

    clear_bm25_cache()

    return {
        "document_id": document_id,
        "deleted": True,
        "chunks_deleted": True,
        "object_deleted": object_deleted,
    }

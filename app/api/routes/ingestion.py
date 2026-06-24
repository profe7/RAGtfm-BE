import hashlib
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.constants import DocumentStatus
from app.db.models import DocumentRecord, UserRecord
from app.db.session import get_db
from app.schemas.ingestion import IngestPdfResponse
from app.services.documents.document_catalog import create_document_record, get_document_by_checksum
from app.services.documents.document_storage import save_uploaded_document
from app.services.ingestion.tasks import process_document_task

settings = get_settings()

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


@router.post(
    "/pdf",
    response_model=IngestPdfResponse,
    responses={
        400: {"description": "Invalid upload. The file must be a readable PDF."},
        409: {"description": "Duplicate file — already uploaded."},
        413: {"description": "PDF file is too large."},
        429: {"description": "Too many documents in the queue for a given user."},
    },
)
async def ingest_pdf(
    file: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="PDF file is too large")

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    existing = get_document_by_checksum(db, user_id=current_user.id, sha256=sha256)
    if existing and existing.status != DocumentStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "This file has already been uploaded.",
                "document_id": existing.document_id,
                "status": existing.status,
            },
        )

    queued_count = (
        db.query(DocumentRecord)
        .filter(
            DocumentRecord.user_id == current_user.id,
            DocumentRecord.status.in_([DocumentStatus.QUEUED, DocumentStatus.PROCESSING]),
        )
        .count()
    )

    if queued_count >= 5:
        raise HTTPException(
            status_code=429,
            detail="You have too many documents in the queue. Please wait for some to finish.",
        )

    document_id = uuid4()

    document_metadata = save_uploaded_document(
        file_bytes=file_bytes,
        document_id=document_id,
        original_filename=file.filename or "uploaded.pdf",
        content_type=file.content_type or "application/pdf",
        sha256=sha256,
        s3_endpoint_url=settings.s3_endpoint_url,
        s3_access_key_id=settings.s3_access_key_id,
        s3_secret_access_key=settings.s3_secret_access_key,
        s3_bucket_name=settings.s3_bucket_name,
        s3_region=settings.s3_region,
        s3_expected_bucket_owner=settings.s3_expected_bucket_owner,
    )

    create_document_record(
        db=db,
        document_metadata=document_metadata,
        chunk_count=0,
        stored_chunk_count=0,
        status=DocumentStatus.QUEUED,
        user_id=current_user.id,
    )

    process_document_task.delay(
        document_id=str(document_id),
        storage_path=document_metadata["storage_path"],
        filename=file.filename or "uploaded.pdf",
        user_id=current_user.id,
    )

    return {
        "document_id": str(document_id),
        "document": document_metadata,
        "filename": file.filename,
        "status": DocumentStatus.QUEUED,
        "chunk_count": 0,
        "stored_chunk_count": 0,
        "stored_chunk_ids": [],
    }

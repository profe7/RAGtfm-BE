import hashlib
from typing import Annotated
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.vectorstores.chroma_store import store_documents
from app.core.config import get_settings
from app.services.documents.document_storage import save_uploaded_document
from fastapi import Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.documents.document_catalog import create_document_record, get_document_by_checksum
from app.schemas.ingestion import IngestPdfResponse
from app.services.ingestion.tasks import process_document_task
from app.api.deps import get_current_user
from app.db.models import UserRecord

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
    if existing and existing.status != "FAILED":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "This file has already been uploaded.",
                "document_id": existing.document_id,
                "status": existing.status,
            },
        )

    document_id = uuid4()

    document_metadata = save_uploaded_document(
        file_bytes=file_bytes,
        document_id=document_id,
        original_filename=file.filename or "uploaded.pdf",
        content_type=file.content_type or "application/pdf",
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
        status="PROCESSING",
        user_id=current_user.id,
        sha256=sha256,
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
        "status": "PROCESSING",
        "chunk_count": 0,
        "stored_chunk_count": 0,
        "stored_chunk_ids": [],
    }


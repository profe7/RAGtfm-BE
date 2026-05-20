from typing import Annotated
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.vectorstores.chroma_store import store_documents
from app.core.config import get_settings
from app.services.documents.document_storage import save_uploaded_document

settings = get_settings()

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


@router.post(
    "/pdf",
    responses={
        400: {
            "description": "Invalid upload. The file must be a readable PDF.",
        },
        413: {
            "description": "PDF file is too large.",
        },
    },
)
async def ingest_pdf(file: Annotated[UploadFile, File()]):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF file is too large",
        )

    try:
        documents = extract_pdf_documents_by_title(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded.pdf",
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read PDF file",
        )

    document_id = uuid4()

    document_metadata = save_uploaded_document(
        file_bytes=file_bytes,
        document_id=document_id,
        original_filename=file.filename or "uploaded.pdf",
        content_type=file.content_type or "application/pdf",
        storage_backend=settings.storage_backend,
        documents_storage_path=settings.documents_storage_path,
    )

    stored_chunk_ids = store_documents(
        document_id=str(document_id),
        documents=documents,
    )

    return {
        "document_id": str(document_id),
        "document": document_metadata,
        "filename": file.filename,
        "chunk_count": len(documents),
        "stored_chunk_count": len(stored_chunk_ids),
        "stored_chunk_ids": stored_chunk_ids,
    }


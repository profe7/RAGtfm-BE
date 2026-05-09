from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.vectorstores.chroma_store import store_documents


router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


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

    document_id = str(uuid4())

    stored_chunk_ids = store_documents(
        document_id=document_id,
        documents=documents,
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunk_count": len(documents),
        "stored_chunk_count": len(stored_chunk_ids),
        "stored_chunk_ids": stored_chunk_ids,
    }


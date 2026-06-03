import logging
from app.db.session import SessionLocal
from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.vectorstores.chroma_store import store_documents
from app.services.documents.document_catalog import update_document_status

logger = logging.getLogger(__name__)

def process_document_task(document_id: str, file_bytes: bytes, filename: str):
    db = SessionLocal()
    try:
        documents = extract_pdf_documents_by_title(
            file_bytes=file_bytes,
            filename=filename,
        )
        
        stored_chunk_ids = store_documents(
            document_id=document_id,
            documents=documents,
        )
        
        update_document_status(
            db=db,
            document_id=document_id,
            status="READY",
            chunk_count=len(documents),
            stored_chunk_count=len(stored_chunk_ids)
        )
        logger.info(f"Successfully processed document {document_id}")
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {str(e)}")
        update_document_status(
            db=db,
            document_id=document_id,
            status="FAILED"
        )
    finally:
        db.close()

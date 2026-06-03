import logging
from app.db.session import SessionLocal
from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.vectorstores.chroma_store import store_documents
from app.services.documents.document_catalog import update_document_status
from app.services.documents.document_storage import download_document_from_s3_storage

logger = logging.getLogger(__name__)
settings = get_settings()

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: str, storage_path: str, filename: str):
    db = SessionLocal()
    try:
        logger.info(f"Downloading {filename} from S3 path: {storage_path}")
        file_bytes = download_document_from_s3_storage(
            storage_path=storage_path,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            bucket_name=settings.s3_bucket_name,
            region=settings.s3_region,
        )

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
        try:
            self.retry(countdown=10 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            update_document_status(
                db=db,
                document_id=document_id,
                status="FAILED"
            )
            raise e
    finally:
        db.close()

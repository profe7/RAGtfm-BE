import logging

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.constants import DocumentStatus
from app.db.session import SessionLocal
from app.services.documents.document_catalog import update_document_status
from app.services.documents.document_storage import download_document_from_s3_storage
from app.services.events.redis_publisher import publish_document_event
from app.services.pdf_loader import extract_pdf_documents_by_title
from app.services.retrieval.bm25_retriever import clear_bm25_cache
from app.services.vectorstores.chroma_store import store_documents
from app.services.vision.image_store import persist_image_chunks

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: str, storage_path: str, filename: str, user_id: str):
    db = SessionLocal()
    try:
        result = update_document_status(
            db=db, document_id=document_id, status=DocumentStatus.PROCESSING
        )
        if result is None:
            logger.warning(f"Document {document_id} no longer exists, aborting task")
            return

        publish_document_event(
            user_id=user_id,
            document_id=document_id,
            status=DocumentStatus.PROCESSING,
        )

        logger.info(f"Downloading {filename} from S3 path: {storage_path}")
        file_bytes = download_document_from_s3_storage(
            storage_path=storage_path,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            bucket_name=settings.s3_bucket_name,
            region=settings.s3_region,
            s3_expected_bucket_owner=settings.s3_expected_bucket_owner,
        )

        documents = extract_pdf_documents_by_title(
            file_bytes=file_bytes,
            filename=filename,
        )

        persist_image_chunks(
            documents=documents,
            document_id=document_id,
            settings=settings,
        )

        stored_chunk_ids = store_documents(
            document_id=document_id,
            documents=documents,
            user_id=user_id,
        )

        update_document_status(
            db=db,
            document_id=document_id,
            status=DocumentStatus.READY,
            chunk_count=len(documents),
            stored_chunk_count=len(stored_chunk_ids),
        )

        clear_bm25_cache()

        publish_document_event(
            user_id=user_id,
            document_id=document_id,
            status=DocumentStatus.READY,
            chunk_count=len(documents),
            stored_chunk_count=len(stored_chunk_ids),
        )

        logger.info(f"Successfully processed document {document_id}")

    except Exception as e:
        logger.exception(f"Failed to process document {document_id}: {e!s}")
        try:
            self.retry(countdown=10 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            update_document_status(
                db=db,
                document_id=document_id,
                status=DocumentStatus.FAILED,
            )
            publish_document_event(
                user_id=user_id,
                document_id=document_id,
                status=DocumentStatus.FAILED,
            )
            raise
    finally:
        db.close()

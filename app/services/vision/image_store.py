import logging

from langchain_core.documents import Document

from app.core.config import Settings
from app.services.documents.document_storage import _get_s3_client
from app.services.vision.ollama_captioner import decode_base64_image

logger = logging.getLogger(__name__)


def _image_object_key(document_id: str, chunk_id: str) -> str:
    return f"images/{document_id}/{chunk_id}.png"


def persist_image_chunks(
    documents: list[Document],
    document_id: str,
    settings: Settings,
) -> None:
    """Upload image-chunk bytes to S3 and record the storage path on each chunk.

    Mirrors the ``f"{document_id}-c{index}"`` chunk-id scheme used by
    ``store_documents`` so keys stay aligned with the stored chunks. The raw
    base64 is dropped from Chroma metadata, so S3 becomes the source of the
    image bytes for generation-time grounding.
    """
    client = _get_s3_client(
        settings.s3_endpoint_url,
        settings.s3_access_key_id,
        settings.s3_secret_access_key,
        settings.s3_region,
    )

    for index, document in enumerate(documents, start=1):
        if document.metadata.get("chunk_type") != "image":
            continue

        image_base64 = document.metadata.get("image_base64")
        if not image_base64:
            continue

        chunk_id = f"{document_id}-c{index}"
        object_key = _image_object_key(document_id, chunk_id)

        put_object_kwargs = {
            "Bucket": settings.s3_bucket_name,
            "Key": object_key,
            "Body": decode_base64_image(image_base64),
            "ContentType": document.metadata.get("image_mime_type") or "image/png",
        }

        if settings.s3_expected_bucket_owner:
            put_object_kwargs["ExpectedBucketOwner"] = settings.s3_expected_bucket_owner

        client.put_object(**put_object_kwargs)

        document.metadata["image_storage_path"] = object_key

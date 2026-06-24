import asyncio
import logging

from app.core.config import get_settings
from app.services.documents.document_storage import download_document_from_s3_storage
from app.services.ollama_client import ollama_async_client

logger = logging.getLogger(__name__)
settings = get_settings()

GENERATION_MODEL = settings.generation_model


SYSTEM_PROMPT = """
You are a careful Retrieval Augmented Generation assistant.

Answer using only the provided context.
Only state a fact if the context explicitly answers the exact thing the user asked.
Do not substitute a related but different fact. If the context only covers a
related-but-different concept, treat the answer as missing.
If the answer is not explicitly in the context, respond exactly: I do not know based on the provided context.
Use one short paragraph unless the user asks for steps or a list.
Cite sources inline using [source: chunk_id].
Preserve exact values and units provided in the context.
Do not repeat the same source citation unnecessarily.
"""


def format_chunk_for_context(chunk: dict) -> str:
    metadata = chunk["metadata"]

    source_label = chunk["chunk_id"]
    filename = metadata.get("filename")
    chunk_type = metadata.get("chunk_type")

    return f"[source: {source_label}]\nfilename: {filename}\ntype: {chunk_type}\n\n{chunk['text']}"


def build_context(chunks: list[dict]) -> str:
    formatted_chunks = [format_chunk_for_context(chunk) for chunk in chunks]

    return "\n\n---\n\n".join(formatted_chunks)


def _load_chunk_image_bytes(chunk: dict) -> bytes | None:
    metadata = chunk["metadata"]

    if metadata.get("chunk_type") != "image":
        return None

    storage_path = metadata.get("image_storage_path")
    if not storage_path:
        return None

    try:
        return download_document_from_s3_storage(
            storage_path=storage_path,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            bucket_name=settings.s3_bucket_name,
            region=settings.s3_region,
            s3_expected_bucket_owner=settings.s3_expected_bucket_owner,
        )
    except Exception:
        logger.exception("Failed to load image for chunk %s", chunk.get("chunk_id"))
        return None


async def collect_chunk_images(chunks: list[dict]) -> list[bytes]:
    if not settings.enable_image_generation:
        return []

    loaded = await asyncio.gather(
        *(asyncio.to_thread(_load_chunk_image_bytes, chunk) for chunk in chunks)
    )

    return [image_bytes for image_bytes in loaded if image_bytes]


async def generate_answer(query: str, chunks: list[dict]):
    context = build_context(chunks)
    images = await collect_chunk_images(chunks)

    user_message: dict = {
        "role": "user",
        "content": (f"Question:\n{query}\n\nContext:\n{context}"),
    }

    if images:
        user_message["images"] = images

    stream = await ollama_async_client.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            user_message,
        ],
        options={
            "temperature": 0,
        },
        stream=True,
        think=False,
    )

    async for chunk in stream:
        content = chunk.message.content
        if content:
            yield content

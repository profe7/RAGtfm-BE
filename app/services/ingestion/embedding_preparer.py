from langchain_core.documents import Document

from app.services.vision.image_ocr import ocr_image_bytes
from app.services.vision.ollama_captioner import (
    caption_image_base64,
    decode_base64_image,
)


def normalize_for_embedding(text: str) -> str:
    return " ".join(text.split()).strip()


def prepare_text_embedding(document: Document) -> None:
    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(document.page_content)


def prepare_table_embedding(document: Document) -> None:
    text_as_html = document.metadata.get("text_as_html")
    table_text = document.page_content

    if table_text == "[Table]":
        table_text = ""

    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(table_text or text_as_html or "")


def prepare_image_embedding(document: Document) -> None:
    image_base64 = document.metadata.get("image_base64")

    ocr_text = ""
    if image_base64:
        ocr_text = ocr_image_bytes(decode_base64_image(image_base64))

    image_caption = caption_image_base64(
        image_base64,
        context_text=document.metadata.get("image_context"),
        ocr_text=ocr_text or None,
    )

    document.metadata["image_caption"] = image_caption
    document.metadata["image_ocr_text"] = ocr_text

    combined = "\n".join(part for part in (image_caption, ocr_text) if part)

    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(combined or document.page_content)


def prepare_embedding_fields(documents: list[Document]) -> list[Document]:
    for document in documents:
        chunk_type = document.metadata["chunk_type"]

        if chunk_type == "image":
            prepare_image_embedding(document)
        elif chunk_type == "table":
            prepare_table_embedding(document)
        else:
            prepare_text_embedding(document)

    return documents

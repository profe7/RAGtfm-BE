from langchain_core.documents import Document

from app.services.vision.ollama_captioner import caption_image_base64


def normalize_for_embedding(text: str) -> str:
    return " ".join(text.split()).strip()


def prepare_text_embedding(document: Document) -> None:
    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(
        document.page_content
    )


def prepare_table_embedding(document: Document) -> None:
    text_as_html = document.metadata.get("text_as_html")
    table_text = document.page_content

    if table_text == "[Table]":
        table_text = ""

    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(
        table_text or text_as_html or ""
    )


def prepare_image_embedding(document: Document) -> None:
    image_caption = caption_image_base64(
        document.metadata.get("image_base64")
    )

    document.metadata["image_caption"] = image_caption
    document.metadata["embedding_content_type"] = "text"
    document.metadata["embedding_text"] = normalize_for_embedding(
        image_caption or document.page_content
    )


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

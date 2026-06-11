import json

import chromadb
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.embeddings.ollama_embedder import embed_document_texts


settings = get_settings()
COLLECTION_NAME = settings.chroma_collection_name


def get_chroma_client():
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )


def get_chroma_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


def serialize_metadata(metadata: dict) -> dict:
    serialized = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if key == "image_base64":
            continue

        if isinstance(value, str | int | float | bool):
            serialized[key] = value
        else:
            serialized[f"{key}_json"] = json.dumps(value, default=str)

    return serialized


def document_text_for_chroma(document: Document) -> str:
    if document.metadata["chunk_type"] == "image":
        return document.metadata.get("image_caption") or document.page_content

    return document.page_content


def store_documents(
    document_id: str,
    documents: list[Document],
    user_id: str,
) -> list[str]:
    collection = get_chroma_collection()

    ids = []
    texts = []
    metadatas = []
    embedding_inputs = []

    for index, document in enumerate(documents, start=1):
        chunk_id = f"{document_id}-c{index}"
        embedding_text = document.metadata.get("embedding_text")

        if not embedding_text:
            continue

        ids.append(chunk_id)
        texts.append(document_text_for_chroma(document))
        embedding_inputs.append(embedding_text)

        metadatas.append(
            serialize_metadata({
                **document.metadata,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "user_id": user_id,
            })
        )

    if not ids:
        return []

    embeddings = embed_document_texts(embedding_inputs)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return ids

def delete_document_chunks(document_id: str) -> None:
    collection = get_chroma_collection()

    collection.delete(
        where={
            "document_id": document_id,
        }
    )

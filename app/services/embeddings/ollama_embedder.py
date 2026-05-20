import ollama

from app.core.config import get_settings

settings = get_settings()

EMBEDDING_MODEL = settings.embedding_model


def embed_document_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[
            f"search_document: {text}"
            for text in texts
        ],
    )

    return response["embeddings"]


def embed_query_text(query: str) -> list[float]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=f"search_query: {query}",
    )

    return response["embeddings"][0]

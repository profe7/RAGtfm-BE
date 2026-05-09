import ollama


EMBEDDING_MODEL = "nomic-embed-text"


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

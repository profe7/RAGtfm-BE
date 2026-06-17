from app.services.embeddings.ollama_embedder import embed_query_text
from app.services.vectorstores.chroma_store import build_chroma_where_filter, get_chroma_collection


def retrieve_relevant_chunks(
    query: str,
    limit: int = 5,
    reference_doc: str | None = None,
    user_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    collection = get_chroma_collection()
    query_embedding = embed_query_text(query)

    where = build_chroma_where_filter(
        reference_doc=reference_doc,
        user_id=user_id,
        document_ids=document_ids,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    chunks = []

    result_ids = results["ids"][0]
    result_documents = results["documents"][0]
    result_metadatas = results["metadatas"][0]
    result_distances = results["distances"][0]

    for index, chunk_id in enumerate(result_ids):
        chunks.append({
            "chunk_id": chunk_id,
            "text": result_documents[index],
            "metadata": result_metadatas[index],
            "distance": result_distances[index],
        })

    return chunks

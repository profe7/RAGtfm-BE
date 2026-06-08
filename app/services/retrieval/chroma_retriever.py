from app.services.embeddings.ollama_embedder import embed_query_text
from app.services.vectorstores.chroma_store import get_chroma_collection


def retrieve_relevant_chunks(
    query: str,
    limit: int = 5,
    reference_doc: str | None = None,
    user_id: str = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    collection = get_chroma_collection()
    query_embedding = embed_query_text(query)

    conditions = []
    if reference_doc:
        conditions.append({
            "filename": reference_doc,
        })
    if user_id:
        conditions.append({
            "user_id": user_id,
        })
    if document_ids:
        conditions.append({"document_id": {"$in": document_ids}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {
            "$and": conditions,
        }

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

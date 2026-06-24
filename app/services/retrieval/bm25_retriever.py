from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.retrieval.retrieval_filter import RetrievalFilter
from app.services.vectorstores.chroma_store import build_chroma_where_filter, get_chroma_collection


def get_all_documents_from_chroma(
    retrieval_filter: RetrievalFilter | None = None,
) -> list[Document]:
    collection = get_chroma_collection()

    where = build_chroma_where_filter(retrieval_filter)

    results = collection.get(where=where, include=["documents", "metadatas"])

    documents = []
    for index, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][index] or {}
        documents.append(
            Document(
                page_content=results["documents"][index],
                metadata={**metadata, "chunk_id": chunk_id},
            )
        )

    return documents


@lru_cache(maxsize=64)
def _cached_bm25_retriever(retrieval_filter: RetrievalFilter | None) -> BM25Retriever | None:
    docs = get_all_documents_from_chroma(retrieval_filter)
    if not docs:
        return None
    return BM25Retriever.from_documents(docs)


def clear_bm25_cache() -> None:
    _cached_bm25_retriever.cache_clear()


def document_to_chunk(document: Document, rank: int) -> dict:
    return {
        "chunk_id": document.metadata["chunk_id"],
        "text": document.page_content,
        "metadata": document.metadata,
        "bm25_rank": rank,
    }


def retrieve_bm25_chunks(
    query: str,
    limit: int = 5,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    retriever = _cached_bm25_retriever(retrieval_filter)

    if retriever is None:
        return []

    retriever.k = limit
    retrieved_documents = retriever.invoke(query)

    return [
        document_to_chunk(document, rank)
        for rank, document in enumerate(retrieved_documents, start=1)
    ]

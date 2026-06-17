from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.vectorstores.chroma_store import get_chroma_collection

def get_all_documents_from_chroma(
    reference_doc: str | None = None,
    user_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[Document]:
    collection = get_chroma_collection()

    conditions = []
    if reference_doc:
        conditions.append({"filename" : reference_doc})
    if user_id:
        conditions.append({"user_id" : user_id})
    if document_ids:
        conditions.append({"document_id" : {"$in" : document_ids}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and" : conditions}

    results = collection.get(where=where, include=["documents", "metadatas"])

    documents = []
    for index, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][index] or {}
        documents.append(
            Document(
                page_content=results["documents"][index],
                metadata={**metadata, "chunk_id" : chunk_id},
            )
        )

    return documents

@lru_cache(maxsize=64)
def _cached_bm25_retriever(
    user_id: str | None,
    document_ids_key : frozenset | None,
    reference_doc: str | None,
) -> tuple[Document, ...]:
    document_ids = list(document_ids_key) if document_ids_key else None
    docs = get_all_documents_from_chroma(
        reference_doc=reference_doc,
        user_id=user_id,
        document_ids=document_ids,
    )
    return tuple(docs)

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
    reference_doc: str | None = None,
    user_id: str = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    document_ids_key = frozenset(document_ids) if document_ids else None
    documents = list(
        _cached_bm25_retriever(
            user_id=user_id,
            document_ids_key=document_ids_key,
            reference_doc=reference_doc,
        )
    )

    if not documents:
        return []

    retriever = BM25Retriever.from_documents(documents)
    retriever.k= limit

    retrieved_documents = retriever.invoke(query)

    return [
        document_to_chunk(document, rank)
        for rank, document in enumerate(retrieved_documents, start=1)
    ]
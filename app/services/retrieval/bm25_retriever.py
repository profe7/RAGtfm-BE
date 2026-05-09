from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.vectorstores.chroma_store import get_chroma_collection


def get_all_documents_from_chroma(reference_doc: str | None = None) -> list[Document]:
    collection = get_chroma_collection()

    where = None

    if reference_doc:
        where = {
            "filename": reference_doc,
        }

    results = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    documents = []

    for index, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][index] or {}

        documents.append(
            Document(
                page_content=results["documents"][index],
                metadata={
                    **metadata,
                    "chunk_id": chunk_id,
                },
            )
        )

    return documents


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
) -> list[dict]:
    documents = get_all_documents_from_chroma(reference_doc=reference_doc)

    if not documents:
        return []

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = limit

    retrieved_documents = retriever.invoke(query)

    return [
        document_to_chunk(document, rank)
        for rank, document in enumerate(retrieved_documents, start=1)
    ]

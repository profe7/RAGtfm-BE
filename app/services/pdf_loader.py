from langchain_core.documents import Document

from app.services.ingestion.embedding_preparer import prepare_embedding_fields
from app.services.ingestion.pdf_chunker import chunk_pdf_elements_by_title
from app.services.ingestion.pdf_partitioner import partition_pdf_elements


def extract_pdf_documents_by_title(file_bytes: bytes, filename: str) -> list[Document]:
    elements = partition_pdf_elements(file_bytes)
    documents = chunk_pdf_elements_by_title(elements, filename)
    return prepare_embedding_fields(documents)

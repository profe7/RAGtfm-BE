import hashlib
from pathlib import Path
from uuid import UUID


def calculate_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_local_document_path(
    documents_storage_path: str,
    document_id: str,
) -> Path:
    return Path(documents_storage_path) / f"{document_id}.pdf"


def build_local_document_uri(document_id: str) -> str:
    return f"local://documents/{document_id}.pdf"


def save_document_to_local_storage(
    file_bytes: bytes,
    document_id: UUID,
    documents_storage_path: str,
) -> dict:
    document_id_value = str(document_id)

    ensure_directory(documents_storage_path)

    document_path = build_local_document_path(
        documents_storage_path=documents_storage_path,
        document_id=document_id_value,
    )

    document_path.write_bytes(file_bytes)

    return {
        "storage_backend": "local",
        "storage_uri": build_local_document_uri(document_id_value),
        "storage_path": str(document_path),
    }


def save_uploaded_document(
    file_bytes: bytes,
    document_id: UUID,
    original_filename: str,
    content_type: str,
    storage_backend: str,
    documents_storage_path: str,
) -> dict:
    if storage_backend != "local":
        raise ValueError(f"Unsupported storage backend: {storage_backend}")

    storage_result = save_document_to_local_storage(
        file_bytes=file_bytes,
        document_id=document_id,
        documents_storage_path=documents_storage_path,
    )

    return {
        "document_id": str(document_id),
        "original_filename": original_filename,
        "content_type": content_type,
        "size_bytes": len(file_bytes),
        "sha256": calculate_sha256(file_bytes),
        **storage_result,
    }
from sqlalchemy.orm import Session

from app.db.models import DocumentRecord


def create_document_record(
    db: Session,
    document_metadata: dict,
    chunk_count: int,
    stored_chunk_count: int,
) -> DocumentRecord:
    document = DocumentRecord(
        document_id=document_metadata["document_id"],
        original_filename=document_metadata["original_filename"],
        content_type=document_metadata["content_type"],
        size_bytes=document_metadata["size_bytes"],
        sha256=document_metadata["sha256"],
        storage_backend=document_metadata["storage_backend"],
        storage_uri=document_metadata["storage_uri"],
        storage_path=document_metadata["storage_path"],
        status="READY",
        chunk_count=chunk_count,
        stored_chunk_count=stored_chunk_count,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def list_document_records(db: Session) -> list[DocumentRecord]:
    return (
        db.query(DocumentRecord)
        .order_by(DocumentRecord.created_at.desc())
        .all()
    )


def get_document_record(
    db: Session,
    document_id: str,
) -> DocumentRecord | None:
    return (
        db.query(DocumentRecord)
        .filter(DocumentRecord.document_id == document_id)
        .first()
    )

def delete_document_record(db: Session, document_id: str) -> DocumentRecord | None:
    document = get_document_record(
        db=db,
        document_id=document_id,
    )

    if document is None:
        return None

    db.delete(document)
    db.commit()

    return document
from sqlalchemy.orm import Session

from app.db.models import DocumentRecord


def create_document_record(
    db: Session,
    document_metadata: dict,
    chunk_count: int,
    stored_chunk_count: int,
    user_id: str,
    status: str = "PROCESSING",
) -> DocumentRecord:
    document = DocumentRecord(
        user_id=user_id,
        document_id=document_metadata["document_id"],
        original_filename=document_metadata["original_filename"],
        content_type=document_metadata["content_type"],
        size_bytes=document_metadata["size_bytes"],
        sha256=document_metadata["sha256"],
        storage_backend=document_metadata["storage_backend"],
        storage_uri=document_metadata["storage_uri"],
        storage_path=document_metadata["storage_path"],
        status=status,
        chunk_count=chunk_count,
        stored_chunk_count=stored_chunk_count,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def list_document_records(
    db: Session,
    user_id: str,
    skip: int = 0,
    limit: int = 10,
) -> tuple[list[DocumentRecord], int]:
    query = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.user_id == user_id)
        .order_by(DocumentRecord.created_at.desc())
    )
    total = query.count()
    documents = query.offset(skip).limit(limit).all()
    return documents, total

def get_document_record(db: Session, document_id: str, user_id: str = None) -> DocumentRecord | None:
    query = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id)
    if user_id:
        query = query.filter(DocumentRecord.user_id == user_id)
    return query.first()

def delete_document_record(db: Session, document_id: str, user_id: str = None) -> DocumentRecord | None:
    document = get_document_record(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None:
        return None

    db.delete(document)
    db.commit()

    return document

def update_document_status(db: Session, document_id: str, status: str, chunk_count: int | None = None, stored_chunk_count: int | None = None, user_id: str = None) -> DocumentRecord | None:
    document = get_document_record(db, document_id, user_id)
    if not document:
        return None
        
    document.status = status
    if chunk_count is not None:
        document.chunk_count = chunk_count
    if stored_chunk_count is not None:
        document.stored_chunk_count = stored_chunk_count
        
    db.commit()
    db.refresh(document)
    return document
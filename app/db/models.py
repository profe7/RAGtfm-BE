from datetime import datetime
from uuid import uuid4
from sqlalchemy import DateTime, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class UserRecord(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DocumentRecord(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False, index=True)

    storage_backend: Mapped[str] = mapped_column(String, nullable=False)
    storage_uri: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[str] = mapped_column(String, nullable=False, default="PROCESSING")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stored_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
import re
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import ConversationRecord, MessageRecord

_conversation_not_found = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Conversation not found",
)


def _owned_conversation(db: Session, user_id: str, conversation_id: str):
    return (
        db.query(ConversationRecord)
        .filter(
            ConversationRecord.id == conversation_id,
            ConversationRecord.user_id == user_id,
        )
        .first()
    )


def _default_title(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    if len(normalized) <= 80:
        return normalized
    return f"{normalized[:77].rstrip()}..."


def get_or_create_conversation(db: Session, user_id: str, conversation_id: str | None) -> str:
    if conversation_id is None:
        conversation = ConversationRecord(user_id=user_id)
        db.add(conversation)
        db.commit()
        return conversation.id

    conversation = _owned_conversation(db, user_id, conversation_id)
    if conversation is None:
        raise _conversation_not_found
    return conversation.id


def load_recent_messages(db: Session, conversation_id: str, limit: int) -> list[dict]:
    rows = (
        db.query(MessageRecord)
        .filter(MessageRecord.conversation_id == conversation_id)
        .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows]


def append_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: list[dict] | None = None,
    metrics: dict | None = None,
    message_status: str = "complete",
) -> MessageRecord:
    conversation = (
        db.query(ConversationRecord).filter(ConversationRecord.id == conversation_id).first()
    )
    if conversation is None:
        raise _conversation_not_found

    message = MessageRecord(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        metrics=metrics,
        status=message_status,
    )
    db.add(message)
    conversation.updated_at = datetime.now(UTC)
    if role == "user" and not conversation.title:
        conversation.title = _default_title(content)
    db.commit()
    db.refresh(message)
    return message


def list_conversations(
    db: Session, user_id: str, skip: int, limit: int
) -> tuple[list[tuple[ConversationRecord, int]], int]:
    base = db.query(ConversationRecord).filter(ConversationRecord.user_id == user_id)
    total = base.count()
    rows = (
        db.query(ConversationRecord, func.count(MessageRecord.id))
        .outerjoin(MessageRecord, MessageRecord.conversation_id == ConversationRecord.id)
        .filter(ConversationRecord.user_id == user_id)
        .group_by(ConversationRecord.id)
        .order_by(ConversationRecord.updated_at.desc(), ConversationRecord.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return rows, total


def get_conversation(
    db: Session, user_id: str, conversation_id: str
) -> tuple[ConversationRecord, list[MessageRecord]]:
    conversation = _owned_conversation(db, user_id, conversation_id)
    if conversation is None:
        raise _conversation_not_found
    messages = (
        db.query(MessageRecord)
        .filter(MessageRecord.conversation_id == conversation_id)
        .order_by(MessageRecord.created_at.asc(), MessageRecord.id.asc())
        .all()
    )
    return conversation, messages


def rename_conversation(
    db: Session, user_id: str, conversation_id: str, title: str
) -> ConversationRecord:
    conversation = _owned_conversation(db, user_id, conversation_id)
    if conversation is None:
        raise _conversation_not_found
    conversation.title = title
    conversation.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(conversation)
    return conversation


def delete_conversation(db: Session, user_id: str, conversation_id: str) -> None:
    conversation = _owned_conversation(db, user_id, conversation_id)
    if conversation is None:
        raise _conversation_not_found
    db.query(MessageRecord).filter(MessageRecord.conversation_id == conversation_id).delete(
        synchronize_session=False
    )
    db.delete(conversation)
    db.commit()

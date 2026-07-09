from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import ConversationRecord, MessageRecord

_conversation_not_found = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Conversation not found",
)


def get_or_create_conversation(db: Session, user_id: str, conversation_id: str | None) -> str:
    if conversation_id is None:
        conversation = ConversationRecord(user_id=user_id)
        db.add(conversation)
        db.commit()
        return conversation.id

    conversation = (
        db.query(ConversationRecord)
        .filter(
            ConversationRecord.id == conversation_id,
            ConversationRecord.user_id == user_id,
        )
        .first()
    )
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


def append_message(db: Session, conversation_id: str, role: str, content: str) -> None:
    db.add(MessageRecord(conversation_id=conversation_id, role=role, content=content))
    db.commit()

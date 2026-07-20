import math
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import UserRecord
from app.db.session import get_db
from app.schemas.conversations import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummaryResponse,
    DeleteConversationResponse,
    RenameConversationRequest,
)
from app.services.conversations.conversation_store import (
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def _summary(conversation, message_count: int) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation.id,
        title=conversation.title,
        message_count=message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("", response_model=ConversationListResponse)
def get_conversations(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    rows, total = list_conversations(db, current_user.id, (page - 1) * page_size, page_size)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, math.ceil(total / page_size)),
        "conversations": [_summary(conversation, count) for conversation, count in rows],
    }


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_detail(
    conversation_id: Annotated[str, Path(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    conversation, messages = get_conversation(db, current_user.id, conversation_id)
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": messages,
    }


@router.patch("/{conversation_id}", response_model=ConversationSummaryResponse)
def patch_conversation(
    conversation_id: Annotated[str, Path(min_length=1)],
    payload: RenameConversationRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    conversation = rename_conversation(db, current_user.id, conversation_id, payload.title)
    _, messages = get_conversation(db, current_user.id, conversation_id)
    return _summary(conversation, len(messages))


@router.delete("/{conversation_id}", response_model=DeleteConversationResponse)
def remove_conversation(
    conversation_id: Annotated[str, Path(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
):
    delete_conversation(db, current_user.id, conversation_id)
    return {"conversation_id": conversation_id, "deleted": True}

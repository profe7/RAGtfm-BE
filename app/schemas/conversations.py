from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.retrieval import RetrievedChunkResponse


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    sources: list[RetrievedChunkResponse] | None = None
    metrics: dict[str, Any] | None = None
    status: str
    created_at: datetime


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    conversations: list[ConversationSummaryResponse]


class ConversationDetailResponse(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse]


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def title_must_contain_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Title must contain text")
        return normalized


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool

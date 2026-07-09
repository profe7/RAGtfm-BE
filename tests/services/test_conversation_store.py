import pytest
from fastapi import HTTPException

from app.db.models import UserRecord
from app.services.conversations.conversation_store import (
    append_message,
    get_or_create_conversation,
    load_recent_messages,
)


def _make_user(db, user_id: str) -> None:
    db.add(UserRecord(id=user_id, email=f"{user_id}@example.com", hashed_password="x"))
    db.commit()


def test_creates_new_conversation_when_id_is_none(db_session):
    _make_user(db_session, "user-1")

    conversation_id = get_or_create_conversation(db_session, "user-1", None)

    assert conversation_id
    assert load_recent_messages(db_session, conversation_id, 6) == []


def test_resumes_own_conversation(db_session):
    _make_user(db_session, "user-1")
    created = get_or_create_conversation(db_session, "user-1", None)

    resumed = get_or_create_conversation(db_session, "user-1", created)

    assert resumed == created


def test_rejects_other_users_conversation(db_session):
    _make_user(db_session, "owner")
    _make_user(db_session, "intruder")
    owned = get_or_create_conversation(db_session, "owner", None)

    with pytest.raises(HTTPException) as exc:
        get_or_create_conversation(db_session, "intruder", owned)

    assert exc.value.status_code == 404


def test_rejects_unknown_conversation_id(db_session):
    _make_user(db_session, "user-1")

    with pytest.raises(HTTPException) as exc:
        get_or_create_conversation(db_session, "user-1", "does-not-exist")

    assert exc.value.status_code == 404


def test_loads_messages_in_chronological_order(db_session):
    _make_user(db_session, "user-1")
    conversation_id = get_or_create_conversation(db_session, "user-1", None)

    append_message(db_session, conversation_id, "user", "first")
    append_message(db_session, conversation_id, "assistant", "second")
    append_message(db_session, conversation_id, "user", "third")

    history = load_recent_messages(db_session, conversation_id, 6)

    assert [m["content"] for m in history] == ["first", "second", "third"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]


def test_load_caps_to_most_recent_n(db_session):
    _make_user(db_session, "user-1")
    conversation_id = get_or_create_conversation(db_session, "user-1", None)

    for i in range(10):
        append_message(db_session, conversation_id, "user", f"msg-{i}")

    history = load_recent_messages(db_session, conversation_id, 3)

    assert [m["content"] for m in history] == ["msg-7", "msg-8", "msg-9"]


def test_messages_are_scoped_per_conversation(db_session):
    _make_user(db_session, "user-1")
    conv_a = get_or_create_conversation(db_session, "user-1", None)
    conv_b = get_or_create_conversation(db_session, "user-1", None)

    append_message(db_session, conv_a, "user", "in-a")
    append_message(db_session, conv_b, "user", "in-b")

    assert [m["content"] for m in load_recent_messages(db_session, conv_a, 6)] == ["in-a"]
    assert [m["content"] for m in load_recent_messages(db_session, conv_b, 6)] == ["in-b"]

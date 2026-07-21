import json

import pytest

from app.api.deps import get_current_user
from app.api.routes import rag as rag_route
from app.db.models import ConversationRecord, MessageRecord, UserRecord
from app.main import app


@pytest.fixture
def mock_user(db_session):
    user = UserRecord(id="user-1", email="test@example.com", hashed_password="x")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_client(client, mock_user):
    def override_get_current_user():
        return mock_user

    app.dependency_overrides[get_current_user] = override_get_current_user
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def stub_pipeline(monkeypatch):
    calls = {"contextualize": [], "generate": []}

    def fake_contextualize(history, query):
        calls["contextualize"].append({"history": history, "query": query})
        return query

    def fake_dense_expansion(query):
        return query

    def fake_retrieve(*args, **kwargs):
        return [{"chunk_id": "c1", "text": "ctx", "metadata": {"filename": "f.pdf"}}]

    async def fake_generate(query, chunks, history=None):
        calls["generate"].append({"query": query, "history": history})
        yield "hello "
        yield "world"

    monkeypatch.setattr(rag_route, "contextualize_query", fake_contextualize)
    monkeypatch.setattr(
        rag_route,
        "expand_query_for_dense_retrieval",
        fake_dense_expansion,
    )
    monkeypatch.setattr(rag_route, "retrieve_hybrid_chunks", fake_retrieve)
    monkeypatch.setattr(rag_route, "generate_answer", fake_generate)
    return calls


def _frames(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_first_turn_emits_conversation_id_and_persists(auth_client, db_session, stub_pipeline):
    response = auth_client.post("/api/v1/rag/query", json={"query": "what is X?"})
    assert response.status_code == 200

    frames = _frames(response)
    conv_frame = frames[0]
    assert conv_frame["type"] == "conversation"
    conversation_id = conv_frame["data"]["conversation_id"]
    assert conversation_id

    assert stub_pipeline["contextualize"][0]["history"] == []
    assert stub_pipeline["generate"][0]["history"] == []

    messages = (
        db_session.query(MessageRecord)
        .filter(MessageRecord.conversation_id == conversation_id)
        .order_by(MessageRecord.created_at)
        .all()
    )
    assert [(m.role, m.content) for m in messages] == [
        ("user", "what is X?"),
        ("assistant", "hello world"),
    ]
    assert messages[1].sources[0]["chunk_id"] == "c1"
    assert messages[1].status == "complete"


def test_second_turn_replays_history(auth_client, db_session, stub_pipeline):
    first = auth_client.post("/api/v1/rag/query", json={"query": "what is X?"})
    conversation_id = _frames(first)[0]["data"]["conversation_id"]

    second = auth_client.post(
        "/api/v1/rag/query",
        json={"query": "what about it?", "conversation_id": conversation_id},
    )
    assert second.status_code == 200
    assert _frames(second)[0]["data"]["conversation_id"] == conversation_id

    replayed = stub_pipeline["generate"][-1]["history"]
    assert [(m["role"], m["content"]) for m in replayed] == [
        ("user", "what is X?"),
        ("assistant", "hello world"),
    ]
    assert stub_pipeline["contextualize"][-1]["history"] == replayed


def test_cannot_post_to_another_users_conversation(auth_client, db_session, stub_pipeline):
    other = ConversationRecord(id="conv-other", user_id="someone-else")
    db_session.add(UserRecord(id="someone-else", email="o@example.com", hashed_password="x"))
    db_session.add(other)
    db_session.commit()

    response = auth_client.post(
        "/api/v1/rag/query",
        json={"query": "leak?", "conversation_id": "conv-other"},
    )
    assert response.status_code == 404

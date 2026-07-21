import pytest

from app.api.deps import get_current_user
from app.db.models import UserRecord
from app.main import app
from app.services.conversations.conversation_store import append_message, get_or_create_conversation


@pytest.fixture
def mock_user(db_session):
    user = UserRecord(id="user-1", email="test@example.com", hashed_password="x")
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def auth_client(client, mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield client
    app.dependency_overrides.clear()


def test_lists_reopens_renames_and_deletes_conversation(auth_client, db_session, mock_user):
    conversation_id = get_or_create_conversation(db_session, mock_user.id, None)
    append_message(db_session, conversation_id, "user", "  Explain   the quarterly report  ")
    append_message(
        db_session,
        conversation_id,
        "assistant",
        "Revenue increased [source: 1].",
        sources=[
            {
                "chunk_id": "doc-1-c1",
                "text": "Revenue increased.",
                "metadata": {"document_id": "doc-1", "filename": "report.pdf"},
                "citation": {
                    "document_id": "doc-1",
                    "filename": "report.pdf",
                    "chunk_type": "text",
                    "page_numbers": [3],
                    "source_locations": [],
                },
            }
        ],
        metrics={"total_ms": 42.0},
    )

    listing = auth_client.get("/api/v1/conversations")
    assert listing.status_code == 200
    summary = listing.json()["conversations"][0]
    assert summary["title"] == "Explain the quarterly report"
    assert summary["message_count"] == 2

    detail = auth_client.get(f"/api/v1/conversations/{conversation_id}")
    assert detail.status_code == 200
    messages = detail.json()["messages"]
    assert messages[1]["sources"][0]["citation"]["page_numbers"] == [3]
    assert messages[1]["metrics"] == {"total_ms": 42.0}

    renamed = auth_client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "  Q2   findings  "},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Q2 findings"

    deleted = auth_client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 200
    assert auth_client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404


def test_conversations_are_tenant_scoped(auth_client, db_session):
    other = UserRecord(id="other", email="other@example.com", hashed_password="x")
    db_session.add(other)
    db_session.commit()
    conversation_id = get_or_create_conversation(db_session, other.id, None)

    assert auth_client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert (
        auth_client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": "Nope"}
        ).status_code
        == 404
    )
    assert auth_client.delete(f"/api/v1/conversations/{conversation_id}").status_code == 404


def test_rejects_blank_title(auth_client, db_session, mock_user):
    conversation_id = get_or_create_conversation(db_session, mock_user.id, None)
    response = auth_client.patch(f"/api/v1/conversations/{conversation_id}", json={"title": "   "})
    assert response.status_code == 422

import pytest

from app.api.deps import get_current_user
from app.api.routes import retrieval as retrieval_route
from app.db.models import UserRecord
from app.main import app


@pytest.fixture
def mock_user(db_session):
    user = UserRecord(id="user-1", email="test@example.com", hashed_password="x")
    db_session.add(user)
    db_session.commit()
    return user


def test_retrieval_requires_authentication(client):
    response = client.get("/api/v1/retrieve/chunks", params={"query": "secret"})
    assert response.status_code == 401


def test_retrieval_is_scoped_to_user_and_documents(client, mock_user, monkeypatch):
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [
            {
                "chunk_id": "doc-1-c1",
                "text": "evidence",
                "metadata": {
                    "document_id": "doc-1",
                    "filename": "report.pdf",
                    "chunk_type": "text",
                    "user_id": "user-1",
                },
            }
        ]

    monkeypatch.setattr(retrieval_route, "retrieve_hybrid_chunks", fake_retrieve)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        response = client.get(
            "/api/v1/retrieve/chunks",
            params=[("query", "revenue"), ("document_ids", "doc-1")],
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["retrieval_filter"].user_id == "user-1"
    assert captured["retrieval_filter"].document_ids == frozenset({"doc-1"})
    assert "user_id" not in response.json()["chunks"][0]["metadata"]

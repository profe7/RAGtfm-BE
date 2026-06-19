import pytest

from app.api.deps import get_current_user
from app.api.routes import events
from app.db.models import UserRecord
from app.main import app


@pytest.fixture
def mock_user(db_session):
    user = UserRecord(id=1, email="test@example.com", hashed_password="hashedpassword")
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


def test_create_event_ticket_returns_ticket(auth_client, monkeypatch):
    async def fake_create(user_id):
        assert user_id == "1"  # str(current_user.id)
        return "ticket-abc"

    monkeypatch.setattr(events, "create_sse_ticket", fake_create)

    response = auth_client.post("/api/v1/documents/events/ticket")

    assert response.status_code == 200
    data = response.json()
    assert data["ticket"] == "ticket-abc"
    assert data["expires_in"] == events.settings.sse_ticket_ttl_seconds


def test_create_event_ticket_requires_auth(client):
    response = client.post("/api/v1/documents/events/ticket")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_events_rejects_invalid_ticket(client, monkeypatch):
    async def fake_consume(ticket):
        return None  # expired, already used, or never issued

    monkeypatch.setattr(events, "consume_sse_ticket", fake_consume)

    response = client.get("/api/v1/documents/events?ticket=bad")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired SSE ticket"


def test_events_requires_ticket(client):
    response = client.get("/api/v1/documents/events")

    assert response.status_code == 422  # missing required query param

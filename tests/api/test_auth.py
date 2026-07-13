from app.db.models import UserRecord


def test_register_user_success(client, db_session):
    response = client.post(
        "/api/v1/auth/register", json={"email": "test@example.com", "password": "securepassword"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"

    user_in_db = db_session.query(UserRecord).filter(UserRecord.email == "test@example.com").first()
    assert user_in_db is not None


def test_register_duplicate_user(client):
    client.post(
        "/api/v1/auth/register", json={"email": "duplicate@example.com", "password": "password123"}
    )

    response = client.post(
        "/api/v1/auth/register", json={"email": "duplicate@example.com", "password": "password123"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


def test_login_returns_bearer_token(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "login@example.com", "password": "securepassword"},
    )

    response = client.post(
        "/api/v1/auth/login",
        data={"username": "login@example.com", "password": "securepassword"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]

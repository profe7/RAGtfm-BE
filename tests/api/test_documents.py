import pytest

from app.db.models import UserRecord, DocumentRecord
from app.api.deps import get_current_user
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

@pytest.fixture
def mock_document(db_session, mock_user):
    document = DocumentRecord(
        document_id="doc1",
        original_filename="test_document.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        sha256="fakehash123",
        storage_backend="minio",
        storage_uri="s3://doc1",
        storage_path="path/to/doc1",
        status="COMPLETED",
        user_id=mock_user.id
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return document

def test_list_documents_empty(auth_client):
    response = auth_client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["documents"] == []

def test_list_documents_with_data(auth_client, mock_document):
    response = auth_client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["documents"][0]["document_id"] == "doc1"
    assert data["documents"][0]["original_filename"] == "test_document.pdf"

def test_get_document_by_id(auth_client, mock_document):
    response = auth_client.get(f"/documents/{mock_document.document_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc1"
    assert data["original_filename"] == "test_document.pdf"

def test_get_document_by_id_not_found(auth_client):
    response = auth_client.get("/documents/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"

def test_unauthenticated_access(client):
    response = client.get("/documents/")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
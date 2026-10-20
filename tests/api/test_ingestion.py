import hashlib
from unittest.mock import Mock

import pytest

from app.api.deps import get_current_user
from app.api.routes import ingestion
from app.db.models import DocumentRecord, UserRecord
from app.main import app
from app.services.documents import document_storage

CSV_CONTENT = b"name,score\nAlice,10\n"


@pytest.fixture
def upload_dependencies(monkeypatch):
    storage = Mock()
    queue = Mock()
    monkeypatch.setattr(document_storage, "_get_s3_client", lambda *args: storage)
    monkeypatch.setattr(ingestion.process_document_task, "delay", queue)
    return storage, queue


@pytest.fixture
def upload_user(db_session):
    user = UserRecord(id="uploader", email="upload@example.com", hashed_password="hash")
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def upload_client(client, upload_user):
    app.dependency_overrides[get_current_user] = lambda: upload_user
    yield client
    app.dependency_overrides.pop(get_current_user, None)


def add_document(db_session, user_id, document_id="existing", status="READY", content=CSV_CONTENT):
    document = DocumentRecord(
        document_id=document_id,
        original_filename="scores.csv",
        content_type="text/csv",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        storage_backend="s3",
        storage_uri=f"s3://bucket/documents/{document_id}.csv",
        storage_path=f"documents/{document_id}.csv",
        status=status,
        user_id=user_id,
    )
    db_session.add(document)
    db_session.commit()
    return document


@pytest.mark.parametrize(
    "media_type",
    ["text/csv", "text/csv; charset=utf-8", "application/csv", "application/vnd.ms-excel"],
)
def test_csv_upload_stores_and_queues_document(
    upload_client, upload_user, upload_dependencies, db_session, media_type
):
    storage, queue = upload_dependencies

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.CSV", CSV_CONTENT, media_type)}
    )

    assert response.status_code == 200
    data = response.json()
    document_id = data["document_id"]
    assert data["status"] == "QUEUED"
    assert data["document"]["content_type"] == "text/csv"
    assert data["filename"] == "scores.CSV"
    assert data["chunk_count"] == data["stored_chunk_count"] == 0
    assert data["stored_chunk_ids"] == []
    assert "storage_path" not in data["document"]
    stored = storage.put_object.call_args.kwargs
    assert stored["Key"] == f"documents/{document_id}.csv"
    assert stored["ContentType"] == "text/csv"
    assert stored["Body"] == CSV_CONTENT
    record = db_session.get(DocumentRecord, document_id)
    assert record.user_id == upload_user.id
    assert record.sha256 == hashlib.sha256(CSV_CONTENT).hexdigest()
    assert record.status == "QUEUED"
    queue.assert_called_once_with(
        document_id=document_id,
        storage_path=f"documents/{document_id}.csv",
        filename="scores.CSV",
        user_id=upload_user.id,
        content_type="text/csv",
    )


def test_pdf_upload_still_stores_and_queues_pdf(upload_client, upload_dependencies):
    storage, queue = upload_dependencies

    response = upload_client.post(
        "/api/v1/ingest/pdf", files={"file": ("report.pdf", b"%PDF-1.7 test", "application/pdf")}
    )

    assert response.status_code == 200
    assert response.json()["document"]["content_type"] == "application/pdf"
    assert storage.put_object.call_args.kwargs["Key"].endswith(".pdf")
    assert queue.call_args.kwargs["content_type"] == "application/pdf"


@pytest.mark.parametrize(
    ("route", "media_type"),
    [("csv", "application/pdf"), ("csv", "application/json"), ("pdf", "text/csv")],
)
def test_upload_rejects_wrong_media_type(upload_client, upload_dependencies, route, media_type):
    storage, queue = upload_dependencies

    response = upload_client.post(
        f"/api/v1/ingest/{route}", files={"file": ("test.csv", CSV_CONTENT, media_type)}
    )

    assert response.status_code == 400
    storage.put_object.assert_not_called()
    queue.assert_not_called()


def test_csv_upload_enforces_size_limit(upload_client, upload_dependencies, monkeypatch):
    storage, queue = upload_dependencies
    monkeypatch.setattr(ingestion, "MAX_FILE_SIZE", len(CSV_CONTENT) - 1)

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "CSV file is too large"
    storage.put_object.assert_not_called()
    queue.assert_not_called()


def test_csv_upload_rejects_duplicate(upload_client, upload_user, upload_dependencies, db_session):
    storage, queue = upload_dependencies
    add_document(db_session, upload_user.id)

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("renamed.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 409
    assert response.json()["detail"]["document_id"] == "existing"
    storage.put_object.assert_not_called()
    queue.assert_not_called()


def test_csv_upload_can_retry_failed_document(
    upload_client, upload_user, upload_dependencies, db_session
):
    add_document(db_session, upload_user.id, status="FAILED")

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 200
    assert response.json()["document_id"] != "existing"
    upload_dependencies[1].assert_called_once()


def test_csv_upload_enforces_queue_limit(
    upload_client, upload_user, upload_dependencies, db_session
):
    for index in range(5):
        add_document(
            db_session,
            upload_user.id,
            document_id=f"queued-{index}",
            status="PROCESSING" if index == 0 else "QUEUED",
            content=f"name\n{index}".encode(),
        )

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 429
    upload_dependencies[0].put_object.assert_not_called()
    upload_dependencies[1].assert_not_called()


def test_csv_duplicate_and_queue_checks_are_tenant_scoped(
    upload_client, upload_dependencies, db_session
):
    db_session.add(UserRecord(id="other", email="other@example.com", hashed_password="hash"))
    db_session.commit()
    for index in range(5):
        add_document(db_session, "other", document_id=f"other-{index}", status="QUEUED")

    response = upload_client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 200
    upload_dependencies[1].assert_called_once()


def test_csv_upload_requires_authentication(client, upload_dependencies):
    response = client.post(
        "/api/v1/ingest/csv", files={"file": ("scores.csv", CSV_CONTENT, "text/csv")}
    )

    assert response.status_code == 401
    upload_dependencies[0].put_object.assert_not_called()
    upload_dependencies[1].assert_not_called()

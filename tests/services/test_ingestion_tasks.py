from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services.ingestion import tasks


def test_permanent_processing_errors_are_not_retried(monkeypatch):
    statuses = []
    events = []

    class FakeDb:
        def close(self):
            pass

    monkeypatch.setattr(tasks, "SessionLocal", FakeDb)
    monkeypatch.setattr(
        tasks,
        "update_document_status",
        lambda **kwargs: statuses.append(kwargs["status"]) or object(),
    )
    monkeypatch.setattr(tasks, "publish_document_event", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(tasks, "download_document_from_s3_storage", lambda **kwargs: b"pdf")
    monkeypatch.setattr(
        tasks,
        "extract_pdf_documents_by_title",
        lambda **kwargs: (_ for _ in ()).throw(TypeError("invalid element text")),
    )

    try:
        tasks.process_document_task.run(
            document_id="doc-1",
            storage_path="documents/doc-1.pdf",
            filename="broken.pdf",
            user_id="user-1",
        )
    except TypeError as error:
        assert str(error) == "invalid element text"
    else:
        raise AssertionError("Expected the permanent processing error to propagate")

    assert statuses == ["PROCESSING", "FAILED"]
    assert events[-1]["status"] == "FAILED"


@pytest.fixture
def processing_dependencies(monkeypatch):
    mocks = {}
    for name in (
        "update_document_status",
        "publish_document_event",
        "download_document_from_s3_storage",
        "persist_image_chunks",
        "store_documents",
        "clear_bm25_cache",
    ):
        mocks[name] = Mock()
        monkeypatch.setattr(tasks, name, mocks[name])
    mocks["db"] = Mock()
    monkeypatch.setattr(tasks, "SessionLocal", lambda: mocks["db"])
    mocks["store_documents"].return_value = ["doc-1-c1", "doc-1-c2"]
    mocks["retry"] = Mock()
    monkeypatch.setattr(tasks.process_document_task, "retry", mocks["retry"])
    return mocks


def test_processes_csv_and_publishes_ready_status(processing_dependencies, monkeypatch):
    mocks = processing_dependencies
    mocks["download_document_from_s3_storage"].return_value = b"name\nAlice\nBob\n"
    pdf_loader = Mock()
    monkeypatch.setattr(tasks, "extract_pdf_documents_by_title", pdf_loader)

    tasks.process_document_task.run(
        document_id="doc-1",
        storage_path="documents/doc-1.csv",
        filename="people.csv",
        user_id="user-1",
        content_type="text/csv",
    )

    stored = mocks["store_documents"].call_args.kwargs
    assert stored["document_id"] == "doc-1"
    assert stored["user_id"] == "user-1"
    assert [doc.page_content for doc in stored["documents"]] == ["name: Alice", "name: Bob"]
    assert [call.kwargs["status"] for call in mocks["update_document_status"].call_args_list] == [
        "PROCESSING",
        "READY",
    ]
    mocks["publish_document_event"].assert_called_with(
        user_id="user-1",
        document_id="doc-1",
        status="READY",
        chunk_count=2,
        stored_chunk_count=2,
    )
    pdf_loader.assert_not_called()
    mocks["persist_image_chunks"].assert_not_called()
    mocks["clear_bm25_cache"].assert_called_once_with()
    mocks["retry"].assert_not_called()
    mocks["db"].close.assert_called_once_with()


def test_invalid_csv_fails_without_retry_or_vector_storage(processing_dependencies):
    mocks = processing_dependencies
    mocks["download_document_from_s3_storage"].return_value = b"name,value\nAlice"

    with pytest.raises(ValueError, match="expected 2"):
        tasks.process_document_task.run(
            document_id="doc-1",
            storage_path="documents/doc-1.csv",
            filename="invalid.csv",
            user_id="user-1",
            content_type="text/csv",
        )

    mocks["publish_document_event"].assert_called_with(
        user_id="user-1", document_id="doc-1", status="FAILED"
    )
    mocks["store_documents"].assert_not_called()
    mocks["retry"].assert_not_called()
    mocks["db"].close.assert_called_once_with()


def test_legacy_pdf_tasks_still_use_pdf_loader(processing_dependencies, monkeypatch):
    mocks = processing_dependencies
    mocks["download_document_from_s3_storage"].return_value = b"pdf"
    documents = [Document(page_content="PDF content", metadata={"chunk_type": "text"})]
    pdf_loader = Mock(return_value=documents)
    monkeypatch.setattr(tasks, "extract_pdf_documents_by_title", pdf_loader)

    tasks.process_document_task.run(
        document_id="doc-1",
        storage_path="documents/doc-1.pdf",
        filename="report.pdf",
        user_id="user-1",
    )

    pdf_loader.assert_called_once_with(file_bytes=b"pdf", filename="report.pdf")
    mocks["persist_image_chunks"].assert_called_once()
    assert mocks["store_documents"].call_args.kwargs["documents"] == documents
    mocks["retry"].assert_not_called()

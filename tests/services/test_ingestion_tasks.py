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

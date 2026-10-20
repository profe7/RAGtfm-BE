from unittest.mock import Mock
from uuid import UUID

import pytest

from app.services.documents import document_storage


@pytest.mark.parametrize(
    ("content_type", "extension"),
    [("application/pdf", "pdf"), ("text/csv", "csv")],
)
def test_stores_original_document_with_correct_extension(monkeypatch, content_type, extension):
    client = Mock()
    monkeypatch.setattr(document_storage, "_get_s3_client", lambda *args: client)
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    result = document_storage.save_uploaded_document(
        file_bytes=b"original content",
        document_id=document_id,
        original_filename="untrusted-name.wrong",
        content_type=content_type,
        sha256="checksum",
        s3_endpoint_url="http://s3",
        s3_access_key_id="key",
        s3_secret_access_key="secret",
        s3_bucket_name="documents",
        s3_region="us-east-1",
        s3_expected_bucket_owner="owner",
    )

    object_key = f"documents/{document_id}.{extension}"
    client.put_object.assert_called_once_with(
        Bucket="documents",
        Key=object_key,
        Body=b"original content",
        ContentType=content_type,
        Metadata={"original_filename": "untrusted-name.wrong"},
        ExpectedBucketOwner="owner",
    )
    assert result["storage_path"] == object_key
    assert result["storage_uri"] == f"s3://documents/{object_key}"
    assert result["size_bytes"] == len(b"original content")
    assert result["sha256"] == "checksum"

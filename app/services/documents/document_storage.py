import hashlib
from uuid import UUID

import boto3


def calculate_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_document_to_s3_storage(
    file_bytes: bytes,
    document_id: UUID,
    original_filename: str,
    content_type: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    region: str,
    s3_expected_bucket_owner: str | None,
) -> dict:
    document_id_value = str(document_id)
    object_key = f"documents/{document_id_value}.pdf"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )

    put_object_kwargs = {
        "Bucket": bucket_name,
        "Key": object_key,
        "Body": file_bytes,
        "ContentType": content_type,
        "Metadata": {
            "original_filename": original_filename,
        },
    }

    if s3_expected_bucket_owner:
        put_object_kwargs["ExpectedBucketOwner"] = s3_expected_bucket_owner

    client.put_object(**put_object_kwargs)

    return {
        "storage_backend": "s3",
        "storage_uri": f"s3://{bucket_name}/{object_key}",
        "storage_path": object_key,
    }


def save_uploaded_document(
    file_bytes: bytes,
    document_id: UUID,
    original_filename: str,
    content_type: str,
    s3_endpoint_url: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_bucket_name: str,
    s3_region: str,
    s3_expected_bucket_owner: str | None,
) -> dict:
    storage_result = save_document_to_s3_storage(
        file_bytes=file_bytes,
        document_id=document_id,
        original_filename=original_filename,
        content_type=content_type,
        endpoint_url=s3_endpoint_url,
        access_key_id=s3_access_key_id,
        secret_access_key=s3_secret_access_key,
        bucket_name=s3_bucket_name,
        region=s3_region,
        s3_expected_bucket_owner=s3_expected_bucket_owner,
    )

    return {
        "document_id": str(document_id),
        "original_filename": original_filename,
        "content_type": content_type,
        "size_bytes": len(file_bytes),
        "sha256": calculate_sha256(file_bytes),
        **storage_result,
    }

def delete_document_from_s3_storage(
    storage_path: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    region: str,
    s3_expected_bucket_owner: str | None,
) -> bool:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )

    delete_object_kwargs = {
        "Bucket": bucket_name,
        "Key": storage_path,
    }

    if s3_expected_bucket_owner:
        delete_object_kwargs["ExpectedBucketOwner"] = s3_expected_bucket_owner

    client.delete_object(**delete_object_kwargs)

    return True


def download_document_from_s3_storage(
    storage_path: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    region: str,
) -> bytes:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )

    response = client.get_object(Bucket=bucket_name, Key=storage_path)
    return response["Body"].read()

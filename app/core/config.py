from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAGtfm API"
    app_version: str = "0.1.0"

    secret_key: str = "supersecretkeywohooyoushouldchangethisinproduction"
    access_token_expire_minutes: int = 60 * 24 * 7

    max_file_size_mb: int = 10

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"
    redis_url: str = "redis://redis:6379/0"

    ollama_base_url: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "gemma4:latest"
    vision_model: str = "gemma4:latest"

    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection_name: str = "rag_documents"

    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_bucket_name: str = "ragtfm-documents"
    s3_region: str = "us-east-1"
    s3_expected_bucket_owner: str | None = None

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
    )

    database_url: str = "postgresql+psycopg://postgres:soiree@postgres:5432/ragtfm"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

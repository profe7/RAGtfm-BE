from functools import lru_cache
from secrets import token_urlsafe

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAGtfm API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    secret_key: str = Field(default_factory=lambda: token_urlsafe(32))
    access_token_expire_minutes: int = 60 * 24 * 7
    sse_ticket_ttl_seconds: int = 30

    max_file_size_mb: int = 10

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"
    redis_url: str = "redis://redis:6379/0"

    ollama_base_url: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "qwen3.5:latest"
    vision_model: str = "qwen3.5:latest"
    vision_context_max_chars: int = 800
    enable_image_generation: bool = True

    conversation_history_turns: int = 6

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

    warm_models_on_startup: bool = True

    rate_limit_enabled: bool = True
    rate_limit_storage_uri: str = "memory://"
    login_rate_limit: str = "10/minute"
    register_rate_limit: str = "20/hour"
    rag_query_rate_limit: str = "30/minute"
    evaluation_rate_limit: str = "2/hour"

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
    )

    database_url: str = "postgresql+psycopg://postgres:soiree@postgres:5432/ragtfm"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

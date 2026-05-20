from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAGtfm API"
    app_version: str = "0.1.0"

    max_file_size_mb: int = 10

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "gemma4:latest"
    vision_model: str = "gemma4:e4b"

    chroma_path: str = "chroma_db"
    chroma_collection_name: str = "rag_documents"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
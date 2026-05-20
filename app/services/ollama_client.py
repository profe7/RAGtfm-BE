import ollama

from app.core.config import get_settings


settings = get_settings()

ollama_client = ollama.Client(
    host=settings.ollama_base_url,
)
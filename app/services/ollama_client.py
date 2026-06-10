import ollama
from ollama import AsyncClient, Client

from app.core.config import get_settings

settings = get_settings()

ollama_client = Client(
    host=settings.ollama_base_url,
)

ollama_async_client = AsyncClient(
    host=settings.ollama_base_url,
)
from app.core.config import get_settings
from app.services.ollama_client import ollama_client

settings = get_settings()

_HYDE_PROMPT = (
    "Write a short passage (2-4 sentences) that would directly answer the following question. "
    "Write only the passage itself — no preamble, no explanation.\n\n"
    "Question: {query}"
)

def rewrite_query_hyde(query: str) -> str:
    response = ollama_client.chat(
        model=settings.generation_model,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        options={"temperature": 0.3},
        think=False,
    )
    return response.message.content.strip()
from app.core.config import get_settings
from app.services.ollama_client import ollama_client

settings = get_settings()

_HYDE_PROMPT = (
    "Write a short passage (2-4 sentences) that would directly answer the following question. "
    "Write only the passage itself — no preamble, no explanation.\n\n"
    "Question: {query}"
)

_CONTEXTUALIZE_PROMPT = (
    "Given the conversation so far and a follow-up question, rewrite the follow-up "
    "as a standalone question that can be understood without the conversation. "
    "Resolve pronouns and references (it, that, this, they, the previous one) to the "
    "concrete thing they refer to. Do NOT answer the question. If the follow-up is "
    "already standalone, return it unchanged. Return only the rewritten question.\n\n"
    "Conversation:\n{history}\n\nFollow-up question: {query}"
)


def _format_history(history: list[dict]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def contextualize_query(history: list[dict], query: str) -> str:
    if not history:
        return query

    response = ollama_client.chat(
        model=settings.generation_model,
        messages=[
            {
                "role": "user",
                "content": _CONTEXTUALIZE_PROMPT.format(
                    history=_format_history(history), query=query
                ),
            }
        ],
        options={"temperature": 0},
        think=False,
    )
    return response.message.content.strip()


def rewrite_query_hyde(query: str) -> str:
    response = ollama_client.chat(
        model=settings.generation_model,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        options={"temperature": 0.3},
        think=False,
    )
    return response.message.content.strip()

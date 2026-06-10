from app.services.ollama_client import ollama_client
from app.services.ollama_client import ollama_async_client
from app.core.config import get_settings

settings = get_settings()

GENERATION_MODEL = settings.generation_model


SYSTEM_PROMPT = """
You are a careful RAG assistant.

Answer using only the provided context.
If the answer is not in the context, say: I do not know based on the provided context.
Use one short paragraph unless the user asks for steps or a list.
Cite sources inline using [source: chunk_id].
Preserve exact values, units, button names, LED colors, and mode names from the context.
Do not repeat the same source citation unnecessarily.
"""


def format_chunk_for_context(chunk: dict) -> str:
    metadata = chunk["metadata"]

    source_label = chunk["chunk_id"]
    filename = metadata.get("filename")
    chunk_type = metadata.get("chunk_type")

    return (
        f"[source: {source_label}]\n"
        f"filename: {filename}\n"
        f"type: {chunk_type}\n\n"
        f"{chunk['text']}"
    )


def build_context(chunks: list[dict]) -> str:
    formatted_chunks = [
        format_chunk_for_context(chunk)
        for chunk in chunks
    ]

    return "\n\n---\n\n".join(formatted_chunks)


async def generate_answer(query: str, chunks: list[dict]):
    context = build_context(chunks)

    stream = await ollama_async_client.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT,
            },
            {
                "role":"user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
        options={
            "temperature":0,
        },
        stream=True,
    )

    async for chunk in stream:
        content = chunk.message.content
        if content:
            yield content

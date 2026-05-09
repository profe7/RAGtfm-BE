import json

import ollama


GENERATION_MODEL = "gemma4:latest"


SYSTEM_PROMPT = """
You are a careful RAG assistant.

Answer the user's question using only the provided context.
If the answer is not in the context, say you do not know.
Cite sources inline using [source: chunk_id].
Be concise, factual, and avoid guessing.
"""


def format_chunk_for_context(chunk: dict) -> str:
    metadata = chunk["metadata"]

    source = {
        "chunk_id": chunk["chunk_id"],
        "filename": metadata.get("filename"),
        "chunk_type": metadata.get("chunk_type"),
        "source_order": metadata.get("source_order_json") or metadata.get("source_order"),
        "source_locations": metadata.get("source_locations_json"),
    }

    return (
        f"Source metadata:\n{json.dumps(source, indent=2)}\n\n"
        f"Content:\n{chunk['text']}"
    )


def build_context(chunks: list[dict]) -> str:
    formatted_chunks = [
        format_chunk_for_context(chunk)
        for chunk in chunks
    ]

    return "\n\n---\n\n".join(formatted_chunks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()

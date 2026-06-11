import base64

from app.core.config import get_settings
from app.services.ollama_client import ollama_client

settings = get_settings()

VISION_MODEL = settings.vision_model

IMAGE_CAPTION_PROMPT = """
Describe this image for retrieval in a document RAG system.

Focus on:
- visible objects, diagrams, charts, tables, or UI elements
- any readable text
- what the image is explaining
- facts a user might search for later

Be concise, specific, and factual.
"""


def decode_base64_image(image_base64: str) -> bytes:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    return base64.b64decode(image_base64)


def caption_image_base64(image_base64: str | None) -> str | None:
    if not image_base64:
        return None

    image_bytes = decode_base64_image(image_base64)

    response = ollama_client.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": IMAGE_CAPTION_PROMPT,
                "images": [image_bytes],
            }
        ],
        options={
            "temperature": 0,
        },
        think=False,
    )

    return response["message"]["content"].strip()

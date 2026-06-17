import base64

from app.core.config import get_settings
from app.services.ollama_client import ollama_client

settings = get_settings()

VISION_MODEL = settings.vision_model

IMAGE_CAPTION_PROMPT = """
You are describing an image embedded in a document, for a retrieval system.
{context_block}{ocr_block}
Use the surrounding context and OCR only to disambiguate what you see. Do not
invent facts that are not visible in the image. Respond in exactly two sections:

VISIBLE TEXT: transcribe verbatim any text, numbers, labels, axis values, or
units readable in the image. If there is none, write "none".
DESCRIPTION: concisely and factually describe the objects, diagrams, charts,
tables, or UI shown, and what the image conveys.
"""


def decode_base64_image(image_base64: str) -> bytes:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    return base64.b64decode(image_base64)


def build_caption_prompt(context_text: str | None, ocr_text: str | None) -> str:
    context_block = ""
    if context_text:
        context_block = (
            f"\nSurrounding document context:\n\"\"\"{context_text}\"\"\"\n"
        )

    ocr_block = ""
    if ocr_text:
        ocr_block = (
            f"\nText detected by OCR (may be imperfect):\n\"\"\"{ocr_text}\"\"\"\n"
        )

    return IMAGE_CAPTION_PROMPT.format(
        context_block=context_block,
        ocr_block=ocr_block,
    )


def caption_image_base64(
    image_base64: str | None,
    context_text: str | None = None,
    ocr_text: str | None = None,
) -> str | None:
    if not image_base64:
        return None

    image_bytes = decode_base64_image(image_base64)

    response = ollama_client.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": build_caption_prompt(context_text, ocr_text),
                "images": [image_bytes],
            }
        ],
        options={
            "temperature": 0,
        },
        think=False,
    )

    return response["message"]["content"].strip()

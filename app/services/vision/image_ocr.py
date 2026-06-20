import logging
from io import BytesIO

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def ocr_image_bytes(image_bytes: bytes | None) -> str:
    if not image_bytes:
        return ""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            text = pytesseract.image_to_string(image)
    except Exception:
        logger.exception("OCR failed for image chunk")
        return ""

    return " ".join(text.split()).strip()

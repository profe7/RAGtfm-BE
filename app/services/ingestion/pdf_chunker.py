from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title

from app.core.config import get_settings
from app.services.ingestion.source_metadata import (
    source_location_from_element,
    source_locations_from_chunk,
    source_order_range,
)

settings = get_settings()


TEXT_ELEMENT_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "UncategorizedText",
    "Header",
    "Footer",
}

CONTEXT_ELEMENT_TYPES = TEXT_ELEMENT_TYPES | {"FigureCaption", "Caption"}

SEPARATE_ELEMENT_TYPES = {
    "Table",
    "Image",
}


def build_image_context(
    elements,
    image_element,
    window: int = 2,
    max_chars: int = settings.vision_context_max_chars,
) -> str:
    try:
        image_index = elements.index(image_element)
    except ValueError:
        return ""

    image_page = getattr(image_element.metadata, "page_number", None)

    def context_text(element) -> str | None:
        if element.category not in CONTEXT_ELEMENT_TYPES:
            return None
        text = str(element).strip()
        return text or None

    before: list[str] = []
    for element in reversed(elements[:image_index]):
        if len(before) >= window:
            break
        page = getattr(element.metadata, "page_number", None)
        if image_page is not None and page is not None and page != image_page:
            continue
        text = context_text(element)
        if text:
            before.append(text)
    before.reverse()

    after: list[str] = []
    for element in elements[image_index + 1 :]:
        if len(after) >= window:
            break
        page = getattr(element.metadata, "page_number", None)
        if image_page is not None and page is not None and page != image_page:
            continue
        text = context_text(element)
        if text:
            after.append(text)

    combined = " ".join(before + after).strip()
    return combined[:max_chars]


def chunk_pdf_elements_by_title(elements, filename: str) -> list[Document]:
    text_elements = []
    separate_elements = []

    for element in elements:
        if element.category in SEPARATE_ELEMENT_TYPES:
            separate_elements.append(element)
        elif element.category in TEXT_ELEMENT_TYPES:
            text_elements.append(element)

    chunked_text_elements = chunk_by_title(
        text_elements,
        max_characters=2000,
        new_after_n_chars=1500,
        combine_text_under_n_chars=1200,
        include_orig_elements=True,
    )

    documents = []

    for chunk_index, chunk in enumerate(chunked_text_elements, start=1):
        text = str(chunk).strip()

        if not text:
            continue

        source_locations = source_locations_from_chunk(chunk)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "chunk_type": "text",
                    "element_type": chunk.category,
                    "source_order": source_order_range(source_locations),
                    "source_locations": source_locations,
                },
            )
        )

    for element in separate_elements:
        metadata = element.metadata
        text = str(element).strip()
        source_location = source_location_from_element(element)

        if element.category == "Table":
            page_content = text or "[Table]"
            extra_metadata = {
                "text_as_html": getattr(metadata, "text_as_html", None),
            }
        elif element.category == "Image":
            page_content = text or "[Image]"
            extra_metadata = {
                "image_base64": getattr(metadata, "image_base64", None),
                "image_mime_type": getattr(metadata, "image_mime_type", None),
                "image_context": build_image_context(elements, element),
            }
        else:
            page_content = text
            extra_metadata = {}

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "filename": filename,
                    "chunk_index": len(documents) + 1,
                    "chunk_type": element.category.lower(),
                    "element_type": element.category,
                    "source_order": {
                        "start": source_location["source_order"],
                        "end": source_location["source_order"],
                    },
                    **extra_metadata,
                    "source_locations": [source_location],
                },
            )
        )

    documents.sort(
        key=lambda document: (
            document.metadata["source_order"]["start"]
            if document.metadata["source_order"]["start"] is not None
            else float("inf")
        )
    )

    return documents

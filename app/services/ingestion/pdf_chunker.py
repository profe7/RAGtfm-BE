from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title

from app.services.ingestion.source_metadata import (
    source_location_from_element,
    source_locations_from_chunk,
    source_order_range,
)


TEXT_ELEMENT_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "UncategorizedText",
    "Header",
    "Footer",
}

SEPARATE_ELEMENT_TYPES = {
    "Table",
    "Image",
}


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
        key=lambda document: document.metadata["source_order"]["start"]
        if document.metadata["source_order"]["start"] is not None
        else float("inf")
    )

    return documents

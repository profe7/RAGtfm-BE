from types import SimpleNamespace

from langchain_core.documents import Document

from app.services.ingestion import embedding_preparer
from app.services.ingestion.pdf_chunker import build_image_context


class FakeElement:
    def __init__(self, category, text, page_number=1):
        self.category = category
        self.text = text
        self.metadata = SimpleNamespace(page_number=page_number)

    def __str__(self):
        return self.text


def test_build_image_context_collects_neighbours_within_window():
    elements = [
        FakeElement("Title", "Section 1"),
        FakeElement("NarrativeText", "Text before the image."),
        FakeElement("Image", ""),
        FakeElement("FigureCaption", "Figure 1: a chart."),
        FakeElement("NarrativeText", "Text after the image."),
    ]
    image = elements[2]

    context = build_image_context(elements, image, window=1)

    assert "Text before the image." in context
    assert "Figure 1: a chart." in context
    assert "Text after the image." not in context
    assert "Section 1" not in context


def test_build_image_context_prefers_same_page():
    elements = [
        FakeElement("NarrativeText", "Other page text", page_number=1),
        FakeElement("Image", "", page_number=2),
        FakeElement("NarrativeText", "Same page text", page_number=2),
    ]
    image = elements[1]

    context = build_image_context(elements, image, window=2)

    assert "Same page text" in context
    assert "Other page text" not in context


def test_build_image_context_truncates_to_max_chars():
    elements = [
        FakeElement("Image", ""),
        FakeElement("NarrativeText", "x" * 500),
    ]
    image = elements[0]

    context = build_image_context(elements, image, window=2, max_chars=100)

    assert len(context) == 100


def test_prepare_image_embedding_combines_caption_and_ocr(monkeypatch):
    monkeypatch.setattr(embedding_preparer, "decode_base64_image", lambda _: b"bytes")
    monkeypatch.setattr(embedding_preparer, "ocr_image_bytes", lambda _: "Speed 42 km/h")
    monkeypatch.setattr(
        embedding_preparer,
        "caption_image_base64",
        lambda image_base64, context_text, ocr_text: "DESCRIPTION: a speedometer",
    )

    document = Document(
        page_content="[Image]",
        metadata={"image_base64": "abc", "image_context": "context"},
    )

    embedding_preparer.prepare_image_embedding(document)

    assert document.metadata["image_caption"] == "DESCRIPTION: a speedometer"
    assert document.metadata["image_ocr_text"] == "Speed 42 km/h"
    embedding_text = document.metadata["embedding_text"]
    assert "a speedometer" in embedding_text
    assert "Speed 42 km/h" in embedding_text

import json

from app.services.retrieval.source_presenter import present_retrieved_chunk


def test_presents_coordinates_and_removes_internal_metadata():
    locations = [
        {
            "element_id": "element-1",
            "source_order": 4,
            "element_type": "NarrativeText",
            "page_number": 2,
            "coordinates": {
                "points": [[10, 20], [110, 20], [110, 60], [10, 60]],
                "layout_width": 612,
                "layout_height": 792,
            },
        }
    ]
    chunk = {
        "chunk_id": "doc-1-c1",
        "text": "Grounded evidence",
        "metadata": {
            "document_id": "doc-1",
            "filename": "report.pdf",
            "chunk_type": "text",
            "user_id": "must-not-leak",
            "image_storage_path": "must-not-leak",
            "source_locations_json": json.dumps(locations),
        },
        "rerank_score": 0.91,
    }

    result = present_retrieved_chunk(chunk)

    assert result["citation"]["document_id"] == "doc-1"
    assert result["citation"]["page_numbers"] == [2]
    assert result["citation"]["source_locations"] == locations
    assert "user_id" not in result["metadata"]
    assert "image_storage_path" not in result["metadata"]
    assert result["rerank_score"] == 0.91


def test_malformed_location_metadata_degrades_to_empty_locations():
    result = present_retrieved_chunk(
        {
            "chunk_id": "c1",
            "text": "text",
            "metadata": {"source_locations_json": "not-json"},
        }
    )

    assert result["citation"]["source_locations"] == []
    assert result["citation"]["page_numbers"] == []

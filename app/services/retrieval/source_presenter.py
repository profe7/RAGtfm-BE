import json
from typing import Any

PUBLIC_METADATA_KEYS = {
    "document_id",
    "filename",
    "chunk_index",
    "chunk_type",
    "element_type",
}


def _json_metadata(metadata: dict[str, Any], key: str, default):
    value = metadata.get(key)
    if value is not None:
        return value

    serialized = metadata.get(f"{key}_json")
    if not isinstance(serialized, str):
        return default

    try:
        return json.loads(serialized)
    except (TypeError, ValueError):
        return default


def _source_locations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    locations = _json_metadata(metadata, "source_locations", [])
    if not isinstance(locations, list):
        return []

    public_locations = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        page_number = location.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            page_number = None
        coordinates = location.get("coordinates")
        if not isinstance(coordinates, dict):
            coordinates = None
        public_locations.append(
            {
                "element_id": location.get("element_id"),
                "source_order": location.get("source_order"),
                "element_type": location.get("element_type"),
                "page_number": page_number,
                "coordinates": coordinates,
            }
        )

    return public_locations


def present_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    locations = _source_locations(metadata)
    page_numbers = sorted(
        {location["page_number"] for location in locations if location["page_number"] is not None}
    )
    public_metadata = {
        key: metadata[key] for key in PUBLIC_METADATA_KEYS if metadata.get(key) is not None
    }

    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "metadata": public_metadata,
        "citation": {
            "document_id": metadata.get("document_id"),
            "filename": metadata.get("filename"),
            "chunk_type": metadata.get("chunk_type"),
            "page_numbers": page_numbers,
            "source_locations": locations,
        },
        **{
            key: chunk.get(key)
            for key in (
                "distance",
                "rrf_score",
                "retrieval_sources",
                "dense_rank",
                "bm25_rank",
                "rerank_score",
                "rerank_rank",
            )
            if chunk.get(key) is not None
        },
    }


def present_retrieved_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [present_retrieved_chunk(chunk) for chunk in chunks]

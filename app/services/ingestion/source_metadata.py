def coordinates_to_dict(coordinates) -> dict | None:
    if coordinates is None:
        return None

    return coordinates.to_dict()


def source_location_from_element(element) -> dict:
    metadata = element.metadata

    return {
        "element_id": element.id,
        "source_order": getattr(metadata, "source_order", None),
        "element_type": element.category,
        "page_number": metadata.page_number,
        "coordinates": coordinates_to_dict(metadata.coordinates),
    }


def source_locations_from_chunk(chunk) -> list[dict]:
    original_elements = getattr(chunk.metadata, "orig_elements", None)

    if not original_elements:
        original_elements = [chunk]

    return [
        source_location_from_element(original_element)
        for original_element in original_elements
    ]


def source_order_range(source_locations: list[dict]) -> dict:
    source_orders = [
        location["source_order"]
        for location in source_locations
        if location["source_order"] is not None
    ]

    if not source_orders:
        return {
            "start": None,
            "end": None,
        }

    return {
        "start": min(source_orders),
        "end": max(source_orders),
    }

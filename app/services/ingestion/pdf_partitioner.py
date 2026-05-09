from io import BytesIO

from unstructured.partition.pdf import partition_pdf


def partition_pdf_elements(file_bytes: bytes):
    elements = partition_pdf(
        file=BytesIO(file_bytes),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        languages=["eng"],
    )

    for source_order, element in enumerate(elements, start=1):
        element.metadata.source_order = source_order

    return elements

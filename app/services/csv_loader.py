import csv
from io import StringIO

from langchain_core.documents import Document

from app.services.ingestion.embedding_preparer import prepare_embedding_fields


def extract_csv_documents(file_bytes: bytes, filename: str) -> list[Document]:
    """Load UTF-8, comma-delimited CSV with a header into one document per data row."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV files must use UTF-8 encoding") from error

    if "\x00" in text:
        raise ValueError("CSV files must not contain null bytes")

    reader = csv.reader(StringIO(text, newline=""), strict=True)
    documents = []
    try:
        headers = next(reader, None)
        if not headers or any(not header.strip() for header in headers):
            raise ValueError("CSV files must have a non-empty header for every column")
        headers = [header.strip() for header in headers]
        if len(set(headers)) != len(headers):
            raise ValueError("CSV column headers must be unique")

        # Row numbers refer to CSV records, including the header, not physical lines.
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} columns; expected {len(headers)}"
                )
            if not any(value.strip() for value in row):
                continue
            documents.append(
                Document(
                    page_content="\n".join(
                        f"{header}: {value}" for header, value in zip(headers, row, strict=True)
                    ),
                    metadata={
                        "filename": filename,
                        "chunk_index": len(documents) + 1,
                        "chunk_type": "table",
                        "element_type": "Table",
                        "csv_row": dict(zip(headers, row, strict=True)),
                        "source_order": {"start": row_number, "end": row_number},
                        "source_locations": [
                            {
                                "element_id": f"csv-row-{row_number}",
                                "source_order": row_number,
                                "element_type": "Table",
                                "row_number": row_number,
                                "page_number": None,
                                "coordinates": None,
                            }
                        ],
                    },
                )
            )
    except csv.Error as error:
        raise ValueError(f"Invalid CSV: {error}") from error

    if not documents:
        raise ValueError("CSV files must contain at least one non-empty data row")
    return prepare_embedding_fields(documents)

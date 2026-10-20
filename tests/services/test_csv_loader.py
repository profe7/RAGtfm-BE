import pytest

from app.services.csv_loader import extract_csv_documents
from app.services.retrieval.source_presenter import present_retrieved_chunk
from app.services.vectorstores.chroma_store import serialize_metadata


def test_csv_rows_preserve_headers_values_and_source_locations():
    documents = extract_csv_documents(b"name,score\nAlice,10\nBob,0\n", "scores.csv")

    assert [doc.page_content for doc in documents] == [
        "name: Alice\nscore: 10",
        "name: Bob\nscore: 0",
    ]
    for index, document in enumerate(documents, start=1):
        assert document.metadata["filename"] == "scores.csv"
        assert document.metadata["chunk_type"] == "table"
        assert document.metadata["chunk_index"] == index
        assert document.metadata["embedding_content_type"] == "text"
        assert document.metadata["csv_row"]["score"] == ("10" if index == 1 else "0")
        assert document.metadata["embedding_text"] == " ".join(document.page_content.split())
        assert document.metadata["source_order"] == {"start": index + 1, "end": index + 1}
        assert document.metadata["source_locations"][0]["row_number"] == index + 1
        assert document.metadata["source_locations"][0]["page_number"] is None


def test_csv_supports_bom_unicode_quotes_commas_and_multiline_fields():
    content = '\ufeffname,notes\r\nJosé,"A comma, a ""quote""\r\nand a new line"\r\n'

    documents = extract_csv_documents(content.encode("utf-8"), "notes.csv")

    assert len(documents) == 1
    assert documents[0].page_content == 'name: José\nnotes: A comma, a "quote"\r\nand a new line'
    assert documents[0].metadata["embedding_text"] == (
        'name: José notes: A comma, a "quote" and a new line'
    )


def test_csv_skips_blank_records_without_losing_row_numbers_or_empty_cells():
    documents = extract_csv_documents(b"name,value\n\n, \nAlice,\nBob,2\n", "sparse.csv")

    assert [doc.page_content for doc in documents] == [
        "name: Alice\nvalue: ",
        "name: Bob\nvalue: 2",
    ]
    assert [doc.metadata["chunk_index"] for doc in documents] == [1, 2]
    assert [doc.metadata["source_locations"][0]["row_number"] for doc in documents] == [4, 5]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "non-empty header"),
        (b"\xef\xbb\xbf", "non-empty header"),
        (b"name,\nAlice,1", "non-empty header"),
        (b"name, name\nAlice,Bob", "unique"),
        (b"name,value\n", "non-empty data row"),
        (b"name,value\n\n,\n", "non-empty data row"),
        (b"name,value\nAlice", "row 2 has 1 columns; expected 2"),
        (b"name,value\nAlice,1,extra", "row 2 has 3 columns; expected 2"),
        (b'name,value\nAlice,"unclosed', "Invalid CSV"),
        (b'name,value\nAlice,"quoted"trailing', "Invalid CSV"),
        (b"name\n\xff", "UTF-8"),
        (b"name\nbad\x00value", "null bytes"),
    ],
)
def test_csv_rejects_invalid_or_empty_input(content, message):
    with pytest.raises(ValueError, match=message):
        extract_csv_documents(content, "invalid.csv")


def test_csv_row_citations_survive_vector_metadata_serialization():
    document = extract_csv_documents(b"name\nAlice\n", "people.csv")[0]
    metadata = serialize_metadata({**document.metadata, "document_id": "doc-1"})

    result = present_retrieved_chunk(
        {"chunk_id": "doc-1-c1", "text": document.page_content, "metadata": metadata}
    )

    assert result["citation"]["filename"] == "people.csv"
    assert result["citation"]["page_numbers"] == []
    assert result["citation"]["source_locations"][0]["row_number"] == 2

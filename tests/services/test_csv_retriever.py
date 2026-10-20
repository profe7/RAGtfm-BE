import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services.retrieval import bm25_retriever, csv_retriever, hybrid_retriever
from app.services.retrieval.csv_retriever import csv_row_values, retrieve_csv_row_chunks
from app.services.retrieval.retrieval_filter import RetrievalFilter
from app.services.vectorstores.chroma_store import serialize_metadata


def race_row(position, track="Australia", document_id="races", structured=True):
    values = {
        "Track": track,
        "Position": str(position),
        "Driver": "Winner" if position == 1 else f"Driver {position}",
        "Starting Grid": "1" if position == 3 else "3",
    }
    metadata = {
        "document_id": document_id,
        "chunk_id": f"{document_id}-{track}-{position}",
        "filename": "RaceResults_2025.csv",
        "chunk_type": "table",
    }
    if structured:
        metadata["csv_row"] = values
    return Document(
        page_content="\n".join(f"{key}: {value}" for key, value in values.items()),
        metadata=metadata,
    )


@pytest.fixture
def planner(monkeypatch):
    client = Mock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps(
                {
                    "lookups": [
                        {
                            "document_id": "races",
                            "filters": [
                                {"column": "Track", "value": "Australia"},
                                {"column": "Position", "value": "1"},
                            ],
                        }
                    ]
                }
            )
        )
    )
    monkeypatch.setattr(csv_retriever, "ollama_client", client)
    return client


@pytest.mark.parametrize("structured", [True, False])
def test_finds_winner_beyond_candidate_limit_and_requires_every_filter(planner, structured):
    documents = [race_row(position, structured=structured) for position in range(2, 45)]
    documents += [race_row(1, track="China", structured=structured)]
    documents += [race_row(1, structured=structured)]

    matches = retrieve_csv_row_chunks("Who won the Australian Grand Prix?", documents, limit=1)

    assert [match["chunk_id"] for match in matches] == ["races-Australia-1"]
    assert matches[0]["csv_exact_match"] is True
    assert "Starting Grid: 3" in matches[0]["text"]
    request = planner.chat.call_args.kwargs
    payload = json.loads(request["messages"][1]["content"])
    assert payload["question"] == "Who won the Australian Grand Prix?"
    assert payload["tables"][0]["columns"][0]["sample_values"] == ["Australia", "China"]
    assert request["format"]["type"] == "object"
    assert request["options"]["temperature"] == 0
    assert "untrusted data" in request["messages"][0]["content"]


def test_structured_cells_preserve_multiline_and_colons_after_storage():
    values = {"Name": "Example", "Notes": "First line\nPosition: 1\nLast line"}
    document = Document(
        page_content="flattened text",
        metadata=serialize_metadata({"chunk_type": "table", "csv_row": values}),
    )

    assert csv_row_values(document) == values


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "{}",
        '{"lookups": [{"document_id": "races", "filters": []}]}',
        '{"lookups": [{"document_id": "another-users-document", '
        '"filters": [{"column": "Position", "value": "1"}]}]}',
        '{"lookups": [{"document_id": "races", '
        '"filters": [{"column": "Invented column", "value": "1"}]}]}',
        '{"lookups": [{"document_id": "races", '
        '"filters": [{"column": "Track", "value": "Austria"}]}]}',
        '{"lookups": [{"document_id": "races", '
        '"filters": [{"column": "Position", "operator": "!=", "value": "1"}]}]}',
        '{"lookups": []}',
    ],
)
def test_invalid_out_of_scope_or_nonmatching_plans_do_not_select_rows(planner, content):
    planner.chat.return_value.message.content = content

    assert retrieve_csv_row_chunks("question", [race_row(1)], limit=5) == []


def test_no_csv_documents_skips_planner(planner):
    assert (
        retrieve_csv_row_chunks(
            "Who won?",
            [
                Document(
                    page_content="A PDF table",
                    metadata={"chunk_type": "table", "filename": "a.pdf"},
                )
            ],
            limit=5,
        )
        == []
    )
    planner.chat.assert_not_called()


def test_planner_outage_falls_back_to_semantic_retrieval(planner):
    planner.chat.side_effect = ConnectionError("offline")
    assert retrieve_csv_row_chunks("Who won?", [race_row(1)], limit=5) == []


def test_accepts_model_wrapping_valid_json_in_markdown(planner):
    planner.chat.return_value.message.content = (
        "```json\n" + planner.chat.return_value.message.content + "\n```"
    )

    matches = retrieve_csv_row_chunks("Who won the Australian Grand Prix?", [race_row(1)], limit=5)

    assert matches[0]["chunk_id"] == "races-Australia-1"


def test_exact_lookup_is_generic_to_columns_and_preserves_literal_cells(planner):
    planner.chat.return_value.message.content = json.dumps(
        {
            "lookups": [
                {
                    "document_id": "catalog",
                    "filters": [{"column": "SKU", "value": "A17"}],
                }
            ]
        }
    )
    document = Document(
        page_content="SKU: A17\nPrice: $12.50",
        metadata={
            "document_id": "catalog",
            "chunk_id": "catalog-1",
            "filename": "catalog.csv",
            "chunk_type": "table",
        },
    )

    matches = retrieve_csv_row_chunks("What is the price of A17?", [document], limit=5)

    assert matches[0]["text"] == "SKU: A17\nPrice: $12.50"


def test_full_retrieval_keeps_winner_when_semantic_search_and_reranker_miss_it(
    planner, monkeypatch
):
    documents = [race_row(position) for position in (3, 4, 5, 8, 10, 1)]
    scope = RetrievalFilter(user_id="user-1", document_ids=["races"])
    loader = Mock(return_value=documents)
    monkeypatch.setattr(bm25_retriever, "get_all_documents_from_chroma", loader)
    bm25_retriever.clear_bm25_cache()
    distractors = [
        {"chunk_id": doc.metadata["chunk_id"], "text": doc.page_content, "metadata": doc.metadata}
        for doc in documents[:-1]
    ]
    monkeypatch.setattr(hybrid_retriever, "retrieve_relevant_chunks", lambda **kwargs: distractors)

    def misleading_reranker(query, chunks, limit):
        return sorted(chunks, key=lambda chunk: bool(chunk.get("csv_exact_match")))[:limit]

    monkeypatch.setattr(hybrid_retriever, "rerank_chunks", misleading_reranker)
    try:
        matches = hybrid_retriever.retrieve_hybrid_chunks(
            "Who won the Australian Grand Prix?",
            limit=1,
            candidate_limit=2,
            retrieval_filter=scope,
        )
    finally:
        bm25_retriever.clear_bm25_cache()

    loader.assert_called_once_with(scope)
    assert len(matches) == 1
    assert matches[0]["chunk_id"] == "races-Australia-1"
    assert "csv" in matches[0]["retrieval_sources"]

from app.services.retrieval import hybrid_retriever
from app.services.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
    retrieve_hybrid_chunks,
)


def chunk(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": f"text-{chunk_id}"}


def test_chunk_in_both_lists_sums_contributions_and_records_both_sources():
    dense = [chunk("a"), chunk("b")]
    bm25 = [chunk("b"), chunk("c")]

    fused = reciprocal_rank_fusion(dense, bm25, limit=5, rrf_k=60)
    by_id = {c["chunk_id"]: c for c in fused}

    assert by_id["b"]["rrf_score"] == 1 / (60 + 2) + 1 / (60 + 1)
    assert by_id["b"]["retrieval_sources"] == ["dense", "bm25"]
    assert by_id["b"]["dense_rank"] == 2
    assert by_id["b"]["bm25_rank"] == 1

    assert by_id["a"]["rrf_score"] == 1 / (60 + 1)
    assert by_id["a"]["retrieval_sources"] == ["dense"]
    assert by_id["c"]["rrf_score"] == 1 / (60 + 2)
    assert by_id["c"]["retrieval_sources"] == ["bm25"]


def test_results_are_ranked_by_rrf_score_descending():
    dense = [chunk("a"), chunk("b")]
    bm25 = [chunk("b"), chunk("c")]

    fused = reciprocal_rank_fusion(dense, bm25, limit=5)
    ids = [c["chunk_id"] for c in fused]

    assert ids == ["b", "a", "c"]
    scores = [c["rrf_score"] for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_limit_truncates_to_top_n():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    bm25 = [chunk("d"), chunk("e")]

    fused = reciprocal_rank_fusion(dense, bm25, limit=2)

    assert len(fused) == 2


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([], [], limit=5) == []


def test_one_empty_leg_still_ranks_the_other():
    dense = [chunk("a"), chunk("b")]

    fused = reciprocal_rank_fusion(dense, [], limit=5)
    ids = [c["chunk_id"] for c in fused]

    assert ids == ["a", "b"]
    assert all(c["retrieval_sources"] == ["dense"] for c in fused)


def test_rrf_k_dampens_top_rank_dominance():
    dense = [chunk("a"), chunk("b")]

    small_k = reciprocal_rank_fusion(dense, [], limit=5, rrf_k=1)
    large_k = reciprocal_rank_fusion(dense, [], limit=5, rrf_k=1000)

    small_gap = small_k[0]["rrf_score"] - small_k[1]["rrf_score"]
    large_gap = large_k[0]["rrf_score"] - large_k[1]["rrf_score"]

    assert large_gap < small_gap


def test_fusion_scores_are_independent_of_leg_ordering():
    list_x = [chunk("a"), chunk("b"), chunk("c")]
    list_y = [chunk("b"), chunk("d")]

    forward = reciprocal_rank_fusion(list_x, list_y, limit=10)
    swapped = reciprocal_rank_fusion(list_y, list_x, limit=10)

    forward_scores = {c["chunk_id"]: c["rrf_score"] for c in forward}
    swapped_scores = {c["chunk_id"]: c["rrf_score"] for c in swapped}
    assert forward_scores == swapped_scores


def test_retrieve_hybrid_chunks_orchestrates_legs_and_records_metrics(monkeypatch):
    calls = {}

    def fake_dense(query, limit, retrieval_filter):
        calls["dense_query"] = query
        return [chunk("a"), chunk("b")]

    def fake_bm25(query, limit, retrieval_filter):
        calls["bm25_query"] = query
        return [chunk("b"), chunk("c")]

    def fake_rerank(query, chunks, limit):
        calls["rerank_query"] = query
        return chunks[:limit]

    monkeypatch.setattr(hybrid_retriever, "retrieve_relevant_chunks", fake_dense)
    monkeypatch.setattr(hybrid_retriever, "retrieve_bm25_chunks", fake_bm25)
    monkeypatch.setattr(hybrid_retriever, "rerank_chunks", fake_rerank)

    metrics: dict = {}
    result = retrieve_hybrid_chunks(
        query="original question",
        dense_query="hypothetical answer",
        limit=2,
        candidate_limit=20,
        metrics=metrics,
    )

    assert calls["dense_query"] == "hypothetical answer"
    assert calls["bm25_query"] == "original question"
    assert calls["rerank_query"] == "original question"

    assert len(result) == 2
    for key in (
        "dense_retrieval_ms",
        "bm25_retrieval_ms",
        "candidate_retrieval_ms",
        "rrf_ms",
        "rerank_ms",
    ):
        assert key in metrics


def test_dense_query_defaults_to_query_when_not_rewritten(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_relevant_chunks",
        lambda query, limit, retrieval_filter: seen.update(dense=query) or [],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "retrieve_bm25_chunks",
        lambda query, limit, retrieval_filter: [],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "rerank_chunks",
        lambda query, chunks, limit: chunks[:limit],
    )

    retrieve_hybrid_chunks(query="just the query", dense_query=None)

    assert seen["dense"] == "just the query"

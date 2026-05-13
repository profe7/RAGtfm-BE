from app.services.retrieval.bm25_retriever import retrieve_bm25_chunks
from app.services.retrieval.chroma_retriever import retrieve_relevant_chunks
from app.services.retrieval.cross_encoder_reranker import rerank_chunks

from app.utils.timing import timed_stage


def reciprocal_rank_fusion(
    dense_chunks: list[dict],
    bm25_chunks: list[dict],
    limit: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    fused = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        chunk_id = chunk["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **chunk,
                "retrieval_sources": [],
                "rrf_score": 0.0,
            }

        fused[chunk_id]["retrieval_sources"].append("dense")
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, chunk in enumerate(bm25_chunks, start=1):
        chunk_id = chunk["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **chunk,
                "retrieval_sources": [],
                "rrf_score": 0.0,
            }

        fused[chunk_id]["retrieval_sources"].append("bm25")
        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda chunk: chunk["rrf_score"],
        reverse=True,
    )

    return ranked[:limit]


def retrieve_hybrid_chunks(
    query: str,
    limit: int = 5,
    candidate_limit: int = 20,
    reference_doc: str | None = None,
    metrics: dict | None = None,
) -> list[dict]:
    if metrics is None:
        metrics = {}

    with timed_stage(metrics, "dense_retrieval_ms"):
        dense_chunks = retrieve_relevant_chunks(
            query=query,
            limit=candidate_limit,
            reference_doc=reference_doc,
        )

    with timed_stage(metrics, "bm25_retrieval_ms"):
        bm25_chunks = retrieve_bm25_chunks(
            query=query,
            limit=candidate_limit,
            reference_doc=reference_doc,
        )

    with timed_stage(metrics, "rrf_ms"):
        fused_candidates = reciprocal_rank_fusion(
            dense_chunks=dense_chunks,
            bm25_chunks=bm25_chunks,
            limit=candidate_limit,
        )

    with timed_stage(metrics, "rerank_ms"):
        return rerank_chunks(
            query=query,
            chunks=fused_candidates,
            limit=limit,
        )



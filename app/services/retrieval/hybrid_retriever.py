from app.services.retrieval.bm25_retriever import retrieve_bm25_chunks
from app.services.retrieval.chroma_retriever import retrieve_relevant_chunks
from app.services.retrieval.cross_encoder_reranker import rerank_chunks
from app.utils.timing import timed_stage


def _accumulate_rrf(
    fused: dict,
    chunks: list[dict],
    source_name: str,
    rrf_k: int,
) -> None:
    for rank, chunk in enumerate(chunks, start=1):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in fused:
            fused[chunk_id] = {**chunk, "retrieval_sources": [], "rrf_score": 0.0}
        fused[chunk_id]["retrieval_sources"].append(source_name)
        fused[chunk_id][f"{source_name}_rank"] = rank
        fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)


def reciprocal_rank_fusion(
    dense_chunks: list[dict],
    bm25_chunks: list[dict],
    limit: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    fused: dict = {}

    _accumulate_rrf(fused, dense_chunks, "dense", rrf_k)
    _accumulate_rrf(fused, bm25_chunks, "bm25", rrf_k)

    ranked = sorted(fused.values(), key=lambda chunk: chunk["rrf_score"], reverse=True)

    return ranked[:limit]


def retrieve_hybrid_chunks(
    query: str,
    dense_query: str | None = None,
    limit: int = 5,
    candidate_limit: int = 20,
    reference_doc: str | None = None,
    metrics: dict | None = None,
    user_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    if metrics is None:
        metrics = {}

    with timed_stage(metrics, "dense_retrieval_ms"):
        dense_chunks = retrieve_relevant_chunks(
            query=dense_query or query,
            limit=candidate_limit,
            reference_doc=reference_doc,
            user_id=user_id,
            document_ids=document_ids,
        )

    with timed_stage(metrics, "bm25_retrieval_ms"):
        bm25_chunks = retrieve_bm25_chunks(
            query=query,
            limit=candidate_limit,
            reference_doc=reference_doc,
            user_id=user_id,
            document_ids=document_ids,
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

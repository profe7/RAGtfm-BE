from functools import lru_cache

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    limit: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    reranker = get_reranker()

    pairs = [
        [query, chunk["text"]]
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    scored_chunks = []

    for chunk, score in zip(chunks, scores):
        scored_chunks.append({
            **chunk,
            "rerank_score": float(score),
        })

    ranked_chunks = sorted(
        scored_chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    for rerank_rank, chunk in enumerate(ranked_chunks, start=1):
        chunk["rerank_rank"] = rerank_rank

    return ranked_chunks[:limit]

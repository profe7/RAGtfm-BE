import re

from app.services.generation.ollama_generator import generate_answer
from app.services.retrieval.hybrid_retriever import retrieve_hybrid_chunks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_term(text: str, term: str) -> bool:
    return normalize_text(term) in normalize_text(text)


def joined_retrieved_text(chunks: list[dict]) -> str:
    return "\n".join(chunk["text"] for chunk in chunks)


def calculate_recall_at_k(chunks: list[dict], ground_truth: list[str]) -> float:
    context = joined_retrieved_text(chunks)

    matched_terms = [
        term
        for term in ground_truth
        if contains_term(context, term)
    ]

    if not ground_truth:
        return 0.0

    return len(matched_terms) / len(ground_truth)


def calculate_hitrate_at_k(chunks: list[dict], ground_truth: list[str]) -> float:
    context = joined_retrieved_text(chunks)

    return float(
        any(contains_term(context, term) for term in ground_truth)
    )


def calculate_mrr(chunks: list[dict], ground_truth: list[str]) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if any(contains_term(chunk["text"], term) for term in ground_truth):
            return 1 / rank

    return 0.0


def calculate_relevance(answer: str, ground_truth: list[str]) -> float:
    matched_terms = [
        term
        for term in ground_truth
        if contains_term(answer, term)
    ]

    if not ground_truth:
        return 0.0

    return len(matched_terms) / len(ground_truth)


def calculate_faithfulness(answer: str, chunks: list[dict]) -> float:
    context = normalize_text(joined_retrieved_text(chunks))
    answer_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", normalize_text(answer)))

    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "is", "are", "was", "were", "it", "this", "that", "with", "by",
    }

    meaningful_words = [
        word
        for word in answer_words
        if word not in stopwords and len(word) > 2
    ]

    if not meaningful_words:
        return 0.0

    supported_words = [
        word
        for word in meaningful_words
        if word in context
    ]

    return len(supported_words) / len(meaningful_words)


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_dataset(dataset: list[dict], k: int = 5) -> dict:
    results = []

    for item in dataset:
        chunks = retrieve_hybrid_chunks(
            query=item["question"],
            limit=k,
            candidate_limit=10,
            reference_doc=item.get("reference_doc"),
        )

        answer = generate_answer(
            query=item["question"],
            chunks=chunks,
        )

        recall_at_k = calculate_recall_at_k(chunks, item["ground_truth"])
        hitrate_at_k = calculate_hitrate_at_k(chunks, item["ground_truth"])
        mrr = calculate_mrr(chunks, item["ground_truth"])
        relevance = calculate_relevance(answer, item["ground_truth"])
        faithfulness = calculate_faithfulness(answer, chunks)

        results.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "answer": answer,
            "reference_doc": item["reference_doc"],
            "ground_truth": item["ground_truth"],
            "metrics": {
                f"recall@{k}": recall_at_k,
                f"hitrate@{k}": hitrate_at_k,
                "mrr": mrr,
                "faithfulness": faithfulness,
                "relevance": relevance,
            },
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "distance": chunk.get("distance"),
                    "rrf_score": chunk.get("rrf_score"),
                    "retrieval_sources": chunk.get("retrieval_sources"),
                    "dense_rank": chunk.get("dense_rank"),
                    "bm25_rank": chunk.get("bm25_rank"),
                    "rerank_score": chunk.get("rerank_score"),
                    "rerank_rank": chunk.get("rerank_rank"),
                }
                for chunk in chunks
            ],
        })

    return {
        "k": k,
        "count": len(results),
        "summary": {
            f"recall@{k}": average([
                result["metrics"][f"recall@{k}"]
                for result in results
            ]),
            f"hitrate@{k}": average([
                result["metrics"][f"hitrate@{k}"]
                for result in results
            ]),
            "mrr": average([
                result["metrics"]["mrr"]
                for result in results
            ]),
            "faithfulness": average([
                result["metrics"]["faithfulness"]
                for result in results
            ]),
            "relevance": average([
                result["metrics"]["relevance"]
                for result in results
            ]),
        },
        "results": results,
    }

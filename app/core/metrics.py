from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

RAG_QUERIES = Counter(
    "rag_queries_total",
    "Total RAG queries served",
)

RAG_STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds",
    "RAG pipeline stage latency in seconds",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

_STAGE_METRIC_KEYS = (
    "query_rewrite_ms",
    "dense_retrieval_ms",
    "bm25_retrieval_ms",
    "candidate_retrieval_ms",
    "rrf_ms",
    "rerank_ms",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
)


def observe_rag_stages(metrics: dict) -> None:
    for key in _STAGE_METRIC_KEYS:
        value = metrics.get(key)
        if value is not None:
            RAG_STAGE_LATENCY.labels(stage=key.removesuffix("_ms")).observe(value / 1000.0)

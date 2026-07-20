from app.core import warmup


def test_models_are_warmed_only_when_both_dependencies_succeed(monkeypatch):
    monkeypatch.setattr(warmup, "get_reranker", lambda: object())
    monkeypatch.setattr(warmup, "embed_query_text", lambda query: [0.1])
    warmup._models_warmed = False

    warmup.warm_models()

    assert warmup.models_warmed() is True


def test_embedder_failure_keeps_models_unready(monkeypatch):
    monkeypatch.setattr(warmup, "get_reranker", lambda: object())

    def fail_embedder(query):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(warmup, "embed_query_text", fail_embedder)
    warmup._models_warmed = False

    warmup.warm_models()

    assert warmup.models_warmed() is False

import structlog

from app.core.config import get_settings
from app.services.embeddings.ollama_embedder import embed_query_text
from app.services.retrieval.cross_encoder_reranker import get_reranker

settings = get_settings()
logger = structlog.get_logger("app.warmup")

_models_warmed = False


def models_warmed() -> bool:
    return _models_warmed


def warm_models() -> None:
    global _models_warmed
    reranker_warmed = False
    embedder_warmed = False

    try:
        get_reranker()
        logger.info("reranker_warmed", model=settings.reranker_model)
        reranker_warmed = True
    except Exception as exc:
        logger.warning("reranker_warm_failed", error=str(exc))

    try:
        embed_query_text("warmup")
        logger.info("embedder_warmed", model=settings.embedding_model)
        embedder_warmed = True
    except Exception as exc:
        logger.warning("embedder_warm_failed", error=str(exc))

    _models_warmed = reranker_warmed and embedder_warmed

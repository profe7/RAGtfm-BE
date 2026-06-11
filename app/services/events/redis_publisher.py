import json

import redis

from app.core.config import get_settings

settings = get_settings()


def publish_document_event(
    user_id: str,
    document_id: str,
    status: str,
    chunk_count: int = 0,
    stored_chunk_count: int = 0,
) -> None:
    r = redis.from_url(settings.redis_url)
    payload = json.dumps({
        "document_id": document_id,
        "status": status,
        "chunk_count": chunk_count,
        "stored_chunk_count": stored_chunk_count,
    })
    r.publish(f"document_events:{user_id}", payload)
    r.close()
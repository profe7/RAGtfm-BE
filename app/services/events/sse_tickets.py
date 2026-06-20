import secrets

import redis.asyncio as aioredis

from app.core.config import get_settings

settings = get_settings()

_TICKET_PREFIX = "sse_ticket:"


async def create_sse_ticket(user_id: str) -> str:
    ticket = secrets.token_urlsafe(32)
    r = aioredis.from_url(settings.redis_url)
    try:
        await r.set(
            f"{_TICKET_PREFIX}{ticket}",
            str(user_id),
            ex=settings.sse_ticket_ttl_seconds,
        )
    finally:
        await r.aclose()
    return ticket


async def consume_sse_ticket(ticket: str) -> str | None:
    r = aioredis.from_url(settings.redis_url)
    try:
        user_id = await r.getdel(f"{_TICKET_PREFIX}{ticket}")
    finally:
        await r.aclose()

    if user_id is None:
        return None
    if isinstance(user_id, bytes):
        user_id = user_id.decode("utf-8")
    return user_id

import secrets

import redis.asyncio as aioredis

from app.core.config import get_settings

settings = get_settings()

_TICKET_PREFIX = "sse_ticket:"


async def create_sse_ticket(user_id: str) -> str:
    """Mint a single-use, short-lived ticket that authorizes one SSE connection.

    EventSource cannot send an Authorization header, so instead of putting the
    long-lived JWT in the URL, the client trades its Bearer token for one of these
    tickets. The ticket is high-entropy, expires within seconds, and is consumed on
    first use, so a leaked URL is worthless almost immediately.
    """
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
    """Atomically validate and invalidate a ticket, returning its user_id or None.

    GETDEL makes the lookup-and-delete a single atomic op, so a ticket can be
    redeemed exactly once even if two connections race on it.
    """
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

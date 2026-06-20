import asyncio

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import UserRecord
from app.services.events.sse_tickets import consume_sse_ticket, create_sse_ticket

router = APIRouter(
    prefix="/documents",
    tags=["Events"],
)

settings = get_settings()


@router.post("/events/ticket")
async def create_event_ticket(
    current_user: UserRecord = Depends(get_current_user),
):
    ticket = await create_sse_ticket(str(current_user.id))
    return {"ticket": ticket, "expires_in": settings.sse_ticket_ttl_seconds}


@router.get("/events")
async def document_events(
    ticket: str = Query(..., description="Single-use SSE ticket from /documents/events/ticket"),
):
    user_id = await consume_sse_ticket(ticket)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired SSE ticket",
        )

    channel = f"document_events:{user_id}"

    async def event_generator():
        r = aioredis.from_url(settings.redis_url, socket_timeout=None, socket_keepalive=True)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    yield {"event": "document_status", "data": data}
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await r.aclose()

    return EventSourceResponse(event_generator())

import asyncio

import jwt
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.db.models import TokenDenylistRecord, UserRecord
from app.db.session import get_db

router = APIRouter(
    prefix="/documents",
    tags=["Events"],
)

settings = get_settings()


def authenticate_token_string(token: str, db: Session) -> UserRecord:
    """Validate a raw token string — used for SSE since EventSource can't set headers."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    jti: str | None = payload.get("jti")
    if jti is not None:
        denied = db.query(TokenDenylistRecord).filter(
            TokenDenylistRecord.jti == jti
        ).first()
        if denied is not None:
            raise credentials_exception

    user = db.query(UserRecord).filter(UserRecord.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user


@router.get("/events")
async def document_events(
    token: str = Query(..., description="JWT access token"),
    db: Session = Depends(get_db),
):
    user = authenticate_token_string(token, db)
    user_id = str(user.id)
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
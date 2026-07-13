import jwt
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.core.config import get_settings
from app.core.security import decode_access_token

settings = get_settings()


def user_or_ip(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        try:
            subject = decode_access_token(token).get("sub")
        except jwt.PyJWTError:
            subject = None
        if subject:
            return f"user:{subject}"
    return get_remote_address(request)


limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    storage_uri=settings.rate_limit_storage_uri,
    swallow_errors=True,
    default_limits=[],
)

import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = structlog.get_logger("app.request")

METRICS_PATH = "/metrics"


def _header_value(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
            await self.app(scope, receive, send)
            return

        request_id = _header_value(scope, b"x-request-id") or str(uuid.uuid4())
        method = scope["method"]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=method, path=scope["path"]
        )

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            template = _route_template(scope)
            REQUEST_COUNT.labels(method, template, str(status_code)).inc()
            REQUEST_LATENCY.labels(method, template).observe(duration)
            logger.info(
                "request_completed",
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()

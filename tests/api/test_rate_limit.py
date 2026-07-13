from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _build_limited_app() -> FastAPI:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri="memory://",
        enabled=True,
    )
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/ping")
    @limiter.limit("2/minute")
    def ping(request: Request):
        return {"ok": True}

    return app


def test_requests_under_limit_pass():
    client = TestClient(_build_limited_app())

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200


def test_request_over_limit_is_rejected():
    client = TestClient(_build_limited_app())

    client.get("/ping")
    client.get("/ping")
    response = client.get("/ping")

    assert response.status_code == 429
    assert "rate limit" in response.text.lower()

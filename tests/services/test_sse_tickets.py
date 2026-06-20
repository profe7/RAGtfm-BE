import asyncio

import pytest

from app.services.events import sse_tickets


class FakeAsyncRedis:
    """Minimal in-memory stand-in for redis.asyncio supporting set/getdel/aclose."""

    def __init__(self, store: dict):
        self._store = store

    async def set(self, key, value, ex=None):
        self._store[key] = value.encode() if isinstance(value, str) else value

    async def getdel(self, key):
        return self._store.pop(key, None)

    async def aclose(self):
        pass


@pytest.fixture
def fake_redis(monkeypatch):
    store: dict = {}
    monkeypatch.setattr(
        sse_tickets.aioredis,
        "from_url",
        lambda *args, **kwargs: FakeAsyncRedis(store),
    )
    return store


def test_create_sse_ticket_stores_user_id(fake_redis):
    ticket = asyncio.run(sse_tickets.create_sse_ticket("user-123"))

    assert ticket
    assert fake_redis[f"sse_ticket:{ticket}"] == b"user-123"


def test_create_sse_ticket_is_unique_per_call(fake_redis):
    first = asyncio.run(sse_tickets.create_sse_ticket("user-123"))
    second = asyncio.run(sse_tickets.create_sse_ticket("user-123"))

    assert first != second


def test_consume_sse_ticket_returns_user_and_is_single_use(fake_redis):
    ticket = asyncio.run(sse_tickets.create_sse_ticket("user-123"))

    first = asyncio.run(sse_tickets.consume_sse_ticket(ticket))
    second = asyncio.run(sse_tickets.consume_sse_ticket(ticket))

    assert first == "user-123"
    assert second is None


def test_consume_unknown_ticket_returns_none(fake_redis):
    assert asyncio.run(sse_tickets.consume_sse_ticket("does-not-exist")) is None

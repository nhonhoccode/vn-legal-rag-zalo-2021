"""Async Redis client wrapper.

Hỗ trợ 2 mode:
- Production: connect tới real Redis qua `redis_url`.
- Test: inject FakeAsyncRedis qua constructor.

Per Decision 13: Redis dùng cho LLM cache + rate limit + **multi-turn session**.
Phase 9 chỉ implement session API; LLM cache + rate limit defer Phase 14 nếu cần.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class AsyncRedisProtocol(Protocol):
    """Protocol để type-hint cả `redis.asyncio.Redis` và `fakeredis.aioredis.FakeRedis`."""

    async def get(self, key: str) -> bytes | None: ...  # noqa: D401
    async def set(self, key: str, value: bytes | str, *, ex: int | None = ...) -> bool: ...
    async def delete(self, *keys: str) -> int: ...
    async def exists(self, *keys: str) -> int: ...
    async def expire(self, key: str, seconds: int) -> bool: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...


def make_redis_client(redis_url: str) -> AsyncRedisProtocol:
    """Build real Redis async client từ URL."""
    from redis.asyncio import Redis

    return Redis.from_url(redis_url, decode_responses=False)


def make_fake_redis() -> AsyncRedisProtocol:
    """In-memory Redis cho tests."""
    from fakeredis import FakeAsyncRedis

    return FakeAsyncRedis()


__all__ = ["AsyncRedisProtocol", "make_fake_redis", "make_redis_client"]

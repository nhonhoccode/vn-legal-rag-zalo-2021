"""LLM response cache trên Redis (Phase 14).

Key: hash(query + top_k + model + context_chunks_hash).
Value: JSON-serialized AskResult.

TTL: 24h (configurable qua REDIS_TTL_LLM_CACHE).

Trade-off: cache hit save 1 LLM call (~3-4s) + chi phí token. Cache miss vẫn chạy
pipeline bình thường. Best với queries phổ biến lặp lại.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.cache.redis_client import AsyncRedisProtocol

logger = logging.getLogger(__name__)


def _cache_key(
    *,
    query: str,
    top_k: int,
    model: str,
    chunk_ids: list[str],
    domain: str | None = None,
) -> str:
    """Stable hash từ tất cả inputs ảnh hưởng output."""
    payload = json.dumps(
        {
            "q": query.strip(),
            "k": top_k,
            "m": model,
            "d": domain or "",
            "c": sorted(chunk_ids),
        },
        ensure_ascii=False,
    )
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"llm_cache:{h[:32]}"


class LLMResponseCache:
    """Redis-backed cache cho LLM responses."""

    def __init__(self, redis: AsyncRedisProtocol, ttl_seconds: int = 86400) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def get(
        self,
        *,
        query: str,
        top_k: int,
        model: str,
        chunk_ids: list[str],
        domain: str | None = None,
    ) -> dict[str, Any] | None:
        key = _cache_key(query=query, top_k=top_k, model=model, chunk_ids=chunk_ids, domain=domain)
        try:
            raw = await self.redis.get(key)
            if not raw:
                return None
            data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            return data if isinstance(data, dict) else None
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM cache get failed: %s", e)
            return None

    async def set(
        self,
        value: dict[str, Any],
        *,
        query: str,
        top_k: int,
        model: str,
        chunk_ids: list[str],
        domain: str | None = None,
    ) -> None:
        key = _cache_key(query=query, top_k=top_k, model=model, chunk_ids=chunk_ids, domain=domain)
        try:
            await self.redis.set(
                key,
                json.dumps(value, ensure_ascii=False),
                ex=self.ttl_seconds,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM cache set failed: %s", e)

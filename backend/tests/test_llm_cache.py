"""Tests cho LLMResponseCache (Phase 14)."""

import pytest

from src.cache.llm_cache import LLMResponseCache, _cache_key
from src.cache.redis_client import make_fake_redis


@pytest.fixture
async def cache():
    redis = make_fake_redis()
    yield LLMResponseCache(redis=redis, ttl_seconds=60)
    await redis.aclose()


def test_cache_key_deterministic():
    k1 = _cache_key(query="hi", top_k=5, model="m", chunk_ids=["a", "b"])
    k2 = _cache_key(query="hi", top_k=5, model="m", chunk_ids=["b", "a"])  # different order
    assert k1 == k2  # sorted chunk_ids


def test_cache_key_differs_with_query():
    k1 = _cache_key(query="hi", top_k=5, model="m", chunk_ids=[])
    k2 = _cache_key(query="hello", top_k=5, model="m", chunk_ids=[])
    assert k1 != k2


def test_cache_key_differs_with_model():
    k1 = _cache_key(query="hi", top_k=5, model="m1", chunk_ids=[])
    k2 = _cache_key(query="hi", top_k=5, model="m2", chunk_ids=[])
    assert k1 != k2


def test_cache_key_differs_with_domain():
    k1 = _cache_key(query="hi", top_k=5, model="m", chunk_ids=[], domain="labor")
    k2 = _cache_key(query="hi", top_k=5, model="m", chunk_ids=[], domain="criminal")
    assert k1 != k2


def test_cache_key_prefix():
    k = _cache_key(query="x", top_k=5, model="m", chunk_ids=[])
    assert k.startswith("llm_cache:")


@pytest.mark.asyncio
async def test_set_get_round_trip(cache):
    await cache.set(
        {"answer": "test answer", "citations": []},
        query="Bồi thường?", top_k=5, model="m", chunk_ids=["a", "b"],
    )
    got = await cache.get(query="Bồi thường?", top_k=5, model="m", chunk_ids=["a", "b"])
    assert got is not None
    assert got["answer"] == "test answer"


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    got = await cache.get(query="never asked", top_k=5, model="m", chunk_ids=[])
    assert got is None


@pytest.mark.asyncio
async def test_cache_preserves_vietnamese(cache):
    await cache.set(
        {"answer": "Người lao động được bồi thường ít nhất 02 tháng."},
        query="q", top_k=5, model="m", chunk_ids=[],
    )
    got = await cache.get(query="q", top_k=5, model="m", chunk_ids=[])
    assert "Người lao động" in got["answer"]
    assert "02 tháng" in got["answer"]


@pytest.mark.asyncio
async def test_different_chunks_different_cache(cache):
    """Same query nhưng retrieved different chunks → different cache entries."""
    await cache.set(
        {"answer": "ans_v1"},
        query="q", top_k=5, model="m", chunk_ids=["a", "b"],
    )
    miss = await cache.get(query="q", top_k=5, model="m", chunk_ids=["c", "d"])
    assert miss is None

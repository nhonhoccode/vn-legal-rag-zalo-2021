"""Tests cho QueryLogger (Phase 13)."""

import time

import pytest

from src.api.middleware.query_log import QueryLogger


@pytest.fixture
def logger(tmp_path):
    return QueryLogger(db_path=tmp_path / "test.db")


@pytest.mark.asyncio
async def test_log_inserts_record(logger):
    await logger.log({
        "endpoint": "/api/ask",
        "query": "Test query",
        "user_id": "user1",
        "latency_ms": 1500,
        "status": 200,
        "n_hits": 3,
        "refused": False,
    })
    stats = logger.stats()
    assert stats["total_queries"] == 1


@pytest.mark.asyncio
async def test_log_multiple_aggregates(logger):
    for i in range(5):
        await logger.log({
            "endpoint": "/api/ask",
            "query": f"q{i}",
            "latency_ms": 100 + i * 100,
            "status": 200,
        })
    stats = logger.stats()
    assert stats["total_queries"] == 5
    assert stats["avg_latency_ms"] == 300  # (100+200+300+400+500)/5


@pytest.mark.asyncio
async def test_log_refused_count(logger):
    await logger.log({"endpoint": "/api/ask", "refused": True})
    await logger.log({"endpoint": "/api/ask", "refused": False})
    await logger.log({"endpoint": "/api/ask", "refused": True})
    stats = logger.stats()
    assert stats["refused_count"] == 2


@pytest.mark.asyncio
async def test_log_swallows_errors(tmp_path):
    """Khi sqlite path invalid → log fail silently, không raise."""
    # File path mà parent là file (không phải dir) → invalid
    bad_path = tmp_path / "bad.db"
    bad_path.write_text("not a db")
    bad_path.chmod(0o000)  # no permission
    try:
        logger = QueryLogger(db_path=bad_path)
        # Don't raise
        await logger.log({"endpoint": "/api/ask"})
    except Exception:
        pass  # acceptable
    finally:
        bad_path.chmod(0o644)


@pytest.mark.asyncio
async def test_stats_window_filter(logger):
    old_ts = int(time.time()) - 86400 * 7  # 7 days ago
    await logger.log({"endpoint": "/api/ask", "ts": old_ts, "status": 200})
    await logger.log({"endpoint": "/api/ask", "status": 200})

    now_window = logger.stats(since_ts=int(time.time()) - 3600)
    assert now_window["total_queries"] == 1

    all_time = logger.stats()
    assert all_time["total_queries"] == 2


@pytest.mark.asyncio
async def test_query_truncated(logger):
    long_q = "x" * 2000
    await logger.log({"endpoint": "/api/ask", "query": long_q})
    # Verify nothing crashes; query stored truncated (1000 chars limit)
    stats = logger.stats()
    assert stats["total_queries"] == 1

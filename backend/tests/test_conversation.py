"""Tests cho SessionManager với fakeredis."""

import pytest

from src.cache.redis_client import make_fake_redis
from src.generation.conversation import SessionManager, Turn


@pytest.fixture
async def manager():
    redis = make_fake_redis()
    m = SessionManager(redis=redis, ttl_seconds=3600, max_turns=6, max_chars=8000)
    yield m
    await redis.aclose()


def test_turn_to_dict_round_trip():
    t = Turn(role="user", content="hello", timestamp=1000, citations=[{"a": 1}])
    d = t.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "hello"
    assert d["timestamp"] == 1000
    assert d["citations"] == [{"a": 1}]

    t2 = Turn.from_dict(d)
    assert t2.role == "user"
    assert t2.content == "hello"


def test_new_session_id_unique():
    a = SessionManager.new_session_id()
    b = SessionManager.new_session_id()
    assert a != b
    assert len(a) == 32  # uuid4 hex


@pytest.mark.asyncio
async def test_session_empty_initially(manager):
    history = await manager.get_history("never-saved")
    assert history == []


@pytest.mark.asyncio
async def test_append_pair_persists(manager):
    sid = SessionManager.new_session_id()
    await manager.append_pair(
        sid,
        user_query="Câu hỏi 1",
        assistant_answer="Trả lời 1",
        citations=[{"law_id": "L1"}],
    )
    history = await manager.get_history(sid)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Câu hỏi 1"
    assert history[1].role == "assistant"
    assert history[1].content == "Trả lời 1"
    assert history[1].citations == [{"law_id": "L1"}]


@pytest.mark.asyncio
async def test_truncate_by_turn_count(manager):
    """max_turns=6 → vượt sẽ drop oldest."""
    sid = SessionManager.new_session_id()
    for i in range(5):
        await manager.append_pair(sid, f"q{i}", f"a{i}")
    # 5 pairs = 10 turns; max 6 → giữ 6 mới nhất
    history = await manager.get_history(sid)
    assert len(history) == 6
    # Last turn = a4
    assert history[-1].content == "a4"


@pytest.mark.asyncio
async def test_truncate_by_char_budget(manager):
    sid = SessionManager.new_session_id()
    long_content = "x" * 5000  # 5000 chars per turn
    # 3 pairs (6 turns × 5000 chars = 30000) → vượt 8000
    for _ in range(3):
        await manager.append_pair(sid, long_content, long_content)
    history = await manager.get_history(sid)
    # Phải bị truncate. Giữ ít nhất 2 turns (rule trong _truncate).
    total = sum(len(t.content) for t in history)
    assert total <= 8000 or len(history) == 2
    assert len(history) >= 2


@pytest.mark.asyncio
async def test_clear_session(manager):
    sid = SessionManager.new_session_id()
    await manager.append_pair(sid, "q", "a")
    assert await manager.exists(sid)
    deleted = await manager.clear(sid)
    assert deleted is True
    assert not await manager.exists(sid)


@pytest.mark.asyncio
async def test_clear_nonexistent_returns_false(manager):
    deleted = await manager.clear("never-existed")
    assert deleted is False


@pytest.mark.asyncio
async def test_append_turn_individually(manager):
    sid = SessionManager.new_session_id()
    await manager.append_turn(sid, Turn(role="user", content="q1"))
    await manager.append_turn(sid, Turn(role="assistant", content="a1"))
    history = await manager.get_history(sid)
    assert len(history) == 2
    assert history[0].role == "user"


@pytest.mark.asyncio
async def test_timestamp_auto_set(manager):
    """Turn với timestamp=0 → tự set khi append."""
    sid = SessionManager.new_session_id()
    await manager.append_turn(sid, Turn(role="user", content="q"))
    history = await manager.get_history(sid)
    assert history[0].timestamp > 0


@pytest.mark.asyncio
async def test_corrupt_session_returns_empty(manager):
    """Nếu data trong Redis corrupt → graceful empty."""
    await manager.redis.set("session:bad", b"not-json{{")
    history = await manager.get_history("bad")
    assert history == []

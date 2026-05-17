"""Tests cho multi-turn RAG pipeline (Phase 9)."""

from unittest.mock import MagicMock

import pytest

from src.cache.redis_client import make_fake_redis
from src.generation.conversation import SessionManager
from src.generation.llm_client import LLMClient, LLMResponse
from src.generation.rag_pipeline import RAGPipeline
from src.vectorstore.base import SearchHit


class RecordingLLM(LLMClient):
    """LLM mock ghi nhận messages received qua từng call."""

    def __init__(self):
        self.calls: list[list] = []
        self.next_answer = "Câu trả lời mặc định."

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return "mock-model"

    async def complete(self, messages, *, max_tokens=1024, temperature=0.0):
        self.calls.append(list(messages))
        return LLMResponse(text=self.next_answer, model="mock-model")

    async def complete_stream(self, messages, *, max_tokens=1024, temperature=0.0):
        self.calls.append(list(messages))
        for w in self.next_answer.split():
            yield w + " "


def _hit(article_id="41"):
    return SearchHit(
        chunk_id=f"l_{article_id}",
        text=f"Điều {article_id} ...",
        score=0.8,
        metadata={"law_id": "45/2019/qh14", "law_title": "BLLĐ", "article_id": article_id},
    )


@pytest.fixture
async def fake_session_manager():
    redis = make_fake_redis()
    yield SessionManager(redis=redis, ttl_seconds=60, max_turns=6, max_chars=8000)
    await redis.aclose()


def _make_retriever(hits):
    m = MagicMock()
    m.search.return_value = hits
    return m


# ============================================================
# Single-turn (regression — không break Phase 8)
# ============================================================
@pytest.mark.asyncio
async def test_single_turn_no_session_id(fake_session_manager):
    retriever = _make_retriever([_hit("41")])
    llm = RecordingLLM()
    llm.next_answer = "[Văn bản 1, Điều 41] OK."
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    result = await pipeline.ask("Câu hỏi đơn?")
    assert result.session_id is None
    assert result.standalone_query is None  # not rewritten
    assert result.metadata["n_history_turns"] == 0
    # System + user, no history
    assert len(llm.calls[0]) == 2


# ============================================================
# Multi-turn
# ============================================================
@pytest.mark.asyncio
async def test_multi_turn_persists_history(fake_session_manager):
    retriever = _make_retriever([_hit("41")])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "test-session-1"
    llm.next_answer = "Trả lời 1."
    await pipeline.ask("Câu 1?", session_id=sid)
    llm.next_answer = "Trả lời 2."
    await pipeline.ask("Câu 2?", session_id=sid)

    history = await fake_session_manager.get_history(sid)
    assert len(history) == 4
    assert history[0].content == "Câu 1?"
    assert history[1].content == "Trả lời 1."
    assert history[2].content == "Câu 2?"
    assert history[3].content == "Trả lời 2."


@pytest.mark.asyncio
async def test_multi_turn_includes_history_in_llm_call(fake_session_manager):
    retriever = _make_retriever([_hit("41")])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "test-session-2"
    await pipeline.ask("First question?", session_id=sid)
    await pipeline.ask("Second question?", session_id=sid)

    # 2nd call should include history.
    # llm.calls[0] = first call (system + user), 2 messages
    # llm.calls[1] = second call (system + 2 history turns + user), 4 messages
    assert len(llm.calls[1]) >= 3  # system + at least 1 history + user


@pytest.mark.asyncio
async def test_rewrite_triggered_for_reference_query(fake_session_manager):
    """Query có 'còn ... thì sao' → trigger rewrite."""
    retriever = _make_retriever([_hit("41")])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "test-rewrite"
    llm.next_answer = "Trả lời câu 1 về bồi thường."
    await pipeline.ask("Sa thải trái pháp luật bồi thường gì?", session_id=sid)

    # Mock LLM trả về rewritten query
    llm.next_answer = "Bồi thường cho người lao động về nghỉ phép thì sao?"
    result = await pipeline.ask("Còn nghỉ phép thì sao?", session_id=sid)

    # Rewrite được trigger → metadata.rewritten=True, standalone_query is set.
    # rewrite_ms có thể = 0 với mock LLM (instant); chỉ check semantic effects.
    assert result.metadata["rewritten"] is True
    assert result.standalone_query is not None
    assert result.standalone_query != "Còn nghỉ phép thì sao?"


@pytest.mark.asyncio
async def test_rewrite_skipped_when_no_history(fake_session_manager):
    retriever = _make_retriever([_hit()])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    result = await pipeline.ask("Còn về vấn đề khác thì sao?", session_id="new-session")
    # Empty history → no rewrite call
    assert result.rewrite_ms == 0


@pytest.mark.asyncio
async def test_rewrite_skipped_when_query_independent(fake_session_manager):
    """Query không chứa reference words → skip rewrite kể cả có history."""
    retriever = _make_retriever([_hit()])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "test-no-rewrite"
    await pipeline.ask("Question 1 standalone?", session_id=sid)
    result = await pipeline.ask("Hợp đồng lao động là gì?", session_id=sid)
    assert result.rewrite_ms == 0


@pytest.mark.asyncio
async def test_session_id_passed_through(fake_session_manager):
    retriever = _make_retriever([_hit()])
    llm = RecordingLLM()
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "specific-session-id"
    result = await pipeline.ask("Test?", session_id=sid)
    assert result.session_id == sid


@pytest.mark.asyncio
async def test_streaming_with_session(fake_session_manager):
    retriever = _make_retriever([_hit()])
    llm = RecordingLLM()
    llm.next_answer = "Streaming answer here."
    pipeline = RAGPipeline(retriever, llm, session_manager=fake_session_manager)

    sid = "stream-session"
    events = []
    async for ev in pipeline.ask_stream("Question?", session_id=sid):
        events.append(ev)

    done = events[-1]
    assert done["type"] == "done"
    assert done["session_id"] == sid

    # History persisted
    history = await fake_session_manager.get_history(sid)
    assert len(history) == 2

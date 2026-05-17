"""Tests cho RAG pipeline với mock LLM + mock retriever."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from src.generation.llm_client import LLMClient, LLMMessage, LLMResponse
from src.generation.rag_pipeline import RAGPipeline
from src.vectorstore.base import SearchHit


class FakeLLM(LLMClient):
    def __init__(self, answer: str = "Câu trả lời mock."):
        self.answer = answer
        self.received_messages: list[LLMMessage] | None = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, messages, *, max_tokens=1024, temperature=0.0):
        self.received_messages = messages
        return LLMResponse(text=self.answer, model="fake-model", prompt_tokens=10, completion_tokens=5)

    async def complete_stream(self, messages, *, max_tokens=1024, temperature=0.0) -> AsyncIterator[str]:
        self.received_messages = messages
        for word in self.answer.split():
            yield word + " "


def _hit(article_id="41", law_id="45/2019/qh14", text="Bồi thường thiệt hại...", score=0.8):
    return SearchHit(
        chunk_id=f"{law_id}_{article_id}",
        text=text,
        score=score,
        metadata={"law_id": law_id, "law_title": "Bộ luật Lao động", "article_id": article_id},
    )


def _make_retriever(hits: list[SearchHit]):
    m = MagicMock()
    m.search.return_value = hits
    return m


@pytest.mark.asyncio
async def test_pipeline_ask_returns_full_result():
    retriever = _make_retriever([_hit(article_id="41")])
    llm = FakeLLM(answer="Theo [Văn bản 1, Điều 41], người lao động được bồi thường.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    result = await pipeline.ask("Bồi thường khi sa thải trái pháp luật?")
    assert result.answer.startswith("Theo")
    assert len(result.citations) == 1
    assert result.citations[0].matched is True
    assert len(result.sources) == 1
    assert result.refused is False
    assert result.model == "fake-model"
    assert result.latency_ms >= 0
    assert result.metadata["n_hits"] == 1
    assert result.metadata["n_citations"] == 1
    assert result.metadata["n_citations_matched"] == 1


@pytest.mark.asyncio
async def test_pipeline_ask_no_hits_marks_refused():
    retriever = _make_retriever([])
    llm = FakeLLM(answer="Tôi không tìm thấy thông tin này.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    result = await pipeline.ask("Câu hỏi out-of-domain")
    assert result.refused is True
    assert len(result.sources) == 0


@pytest.mark.asyncio
async def test_pipeline_llm_refuse_detected():
    retriever = _make_retriever([_hit()])
    llm = FakeLLM(answer="Tôi không tìm thấy thông tin này trong các văn bản đã được cung cấp.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    result = await pipeline.ask("Hỏi gì đó?")
    assert result.refused is True


@pytest.mark.asyncio
async def test_pipeline_passes_query_to_messages():
    retriever = _make_retriever([_hit()])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    await pipeline.ask("My specific query here")
    assert llm.received_messages is not None
    # System prompt + user message
    assert len(llm.received_messages) == 2
    assert "My specific query here" in llm.received_messages[1].content


@pytest.mark.asyncio
async def test_pipeline_domain_filter():
    retriever = _make_retriever([_hit()])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    await pipeline.ask("test", domain="labor")
    # Verify retriever called with where filter
    retriever.search.assert_called_once()
    kwargs = retriever.search.call_args.kwargs
    assert kwargs["where"] == {"domain": "labor"}


@pytest.mark.asyncio
async def test_pipeline_stream_yields_events():
    retriever = _make_retriever([_hit()])
    llm = FakeLLM(answer="Theo Điều 41 thì lao động được bồi thường.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    events = []
    async for ev in pipeline.ask_stream("query"):
        events.append(ev)

    # First event = sources
    assert events[0]["type"] == "sources"
    assert len(events[0]["sources"]) == 1

    # Token events
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 0

    # Last event = done
    assert events[-1]["type"] == "done"
    assert "citations" in events[-1]
    assert "metadata" not in events[-1]  # uses inline fields
    assert "model" in events[-1]


@pytest.mark.asyncio
async def test_pipeline_top_score_low_confidence():
    retriever = _make_retriever([_hit(score=0.1)])  # very low
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, min_top_score=0.3)

    result = await pipeline.ask("test")
    assert result.metadata["low_confidence"] is True


@pytest.mark.asyncio
async def test_pipeline_top_score_high_confidence():
    retriever = _make_retriever([_hit(score=0.8)])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, min_top_score=0.3)

    result = await pipeline.ask("test")
    assert result.metadata["low_confidence"] is False

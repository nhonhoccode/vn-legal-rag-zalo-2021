"""Tests cho synthetic Q&A generator."""

import json
from pathlib import Path

import pytest

from src.eval.synthetic_qa import (
    SyntheticQAItem,
    _parse_questions,
    generate_synthetic_qa_set,
    save_synthetic_qa_to_jsonl,
)
from src.generation.llm_client import LLMClient, LLMResponse


class FakeQALLM(LLMClient):
    def __init__(self, answer: str):
        self.answer = answer

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, messages, *, max_tokens=300, temperature=0.7):
        return LLMResponse(text=self.answer, model="fake-model")

    async def complete_stream(self, messages, *, max_tokens=300, temperature=0.7):
        yield self.answer


def test_parse_questions_numbered():
    text = "1. Câu hỏi 1?\n2. Câu hỏi 2?"
    out = _parse_questions(text)
    assert len(out) == 2
    assert out[0] == "Câu hỏi 1?"
    assert out[1] == "Câu hỏi 2?"


def test_parse_questions_with_paren():
    text = "1) Câu A?\n2) Câu B?"
    out = _parse_questions(text)
    assert len(out) == 2


def test_parse_questions_limits_to_2():
    text = "1. Câu hỏi 1?\n2. Câu hỏi 2?\n3. Câu hỏi 3?\n4. Câu hỏi 4?"
    out = _parse_questions(text)
    assert len(out) == 2


def test_parse_questions_filters_short():
    text = "1. A?\n2. Đây là câu hỏi dài hơn 10 chars?"
    out = _parse_questions(text)
    assert len(out) == 1
    assert out[0].startswith("Đây là")


@pytest.mark.asyncio
async def test_generate_qa_for_chunk_skips_short_text():
    from src.eval.synthetic_qa import generate_qa_for_chunk

    llm = FakeQALLM("1. Câu hỏi 1?")
    items = await generate_qa_for_chunk({"text": "short", "law_id": "L1", "article_id": "1"}, llm)
    assert items == []


@pytest.mark.asyncio
async def test_generate_synthetic_qa_set():
    llm = FakeQALLM(
        "1. Người lao động bị sa thải có được bồi thường không?\n"
        "2. Mức bồi thường tối thiểu khi đơn phương chấm dứt hợp đồng là gì?"
    )
    chunks = [
        {
            "text": "Điều 41. Bồi thường thiệt hại " + "x" * 50,
            "law_id": "45/2019/qh14",
            "article_id": "41",
        },
        {
            "text": "Điều 13. Hợp đồng lao động " + "x" * 50,
            "law_id": "45/2019/qh14",
            "article_id": "13",
        },
    ]
    items = await generate_synthetic_qa_set(chunks, llm, n_samples=2)
    # 2 chunks × 2 questions = 4 items
    assert len(items) == 4
    assert all(isinstance(i, SyntheticQAItem) for i in items)
    # Phrasing alternates
    phrasings = {i.phrasing for i in items}
    assert phrasings == {"natural", "formal"}


def test_save_synthetic_qa_to_jsonl_format(tmp_path: Path):
    items = [
        SyntheticQAItem(
            question_id="syn_0_0",
            question="Câu hỏi 1?",
            law_id="L1",
            article_id="A1",
            source_chunk_text="text...",
        )
    ]
    out = tmp_path / "syn.jsonl"
    n = save_synthetic_qa_to_jsonl(items, out)
    assert n == 1

    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["question_id"] == "syn_0_0"
    assert record["question"] == "Câu hỏi 1?"
    assert record["source"] == "synthetic"
    assert record["relevant_articles"] == [{"law_id": "L1", "article_id": "A1"}]

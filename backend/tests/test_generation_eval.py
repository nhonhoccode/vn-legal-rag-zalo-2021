"""Tests cho generation eval với mock LLM judge."""

import pytest

from src.eval.generation_eval import (
    GenerationSample,
    _parse_json_response,
    evaluate_generation,
    judge_faithfulness,
    judge_relevancy,
)
from src.generation.llm_client import LLMClient, LLMResponse


class FakeJudge(LLMClient):
    """LLM returns canned JSON response."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    @property
    def name(self) -> str:
        return "fake-judge"

    @property
    def model(self) -> str:
        return "fake-judge-model"

    async def complete(self, messages, *, max_tokens=512, temperature=0.0):
        return LLMResponse(text=self.response_text, model="fake-judge-model")

    async def complete_stream(self, messages, *, max_tokens=512, temperature=0.0):
        yield self.response_text


def test_parse_json_response_direct():
    assert _parse_json_response('{"score": 0.8}') == {"score": 0.8}


def test_parse_json_response_with_wrapper():
    text = 'Here is my answer:\n```json\n{"score": 0.5}\n```'
    parsed = _parse_json_response(text)
    assert parsed.get("score") == 0.5


def test_parse_json_invalid_returns_empty():
    assert _parse_json_response("not json at all") == {}


@pytest.mark.asyncio
async def test_judge_faithfulness_perfect():
    judge = FakeJudge('{"score": 1.0, "unsupported_claims": [], "reason": "all good"}')
    sample = GenerationSample(question="q?", answer="a", contexts=["ctx"])
    score, unsupported, reason = await judge_faithfulness(sample, judge)
    assert score == 1.0
    assert unsupported == []
    assert reason == "all good"


@pytest.mark.asyncio
async def test_judge_faithfulness_bad_json():
    judge = FakeJudge("not json")
    sample = GenerationSample(question="q?", answer="a", contexts=[])
    score, unsupported, reason = await judge_faithfulness(sample, judge)
    assert score == 0.0


@pytest.mark.asyncio
async def test_judge_clamps_score():
    judge = FakeJudge('{"score": 1.5}')
    sample = GenerationSample(question="q?", answer="a", contexts=[])
    score, _, _ = await judge_faithfulness(sample, judge)
    assert score == 1.0  # clamped to [0, 1]


@pytest.mark.asyncio
async def test_judge_relevancy():
    judge = FakeJudge('{"score": 0.8, "reason": "mostly relevant"}')
    sample = GenerationSample(question="q?", answer="a", contexts=[])
    score, reason = await judge_relevancy(sample, judge)
    assert score == 0.8
    assert "relevant" in reason


@pytest.mark.asyncio
async def test_evaluate_generation_aggregates():
    judge = FakeJudge('{"score": 0.9}')
    samples = [
        GenerationSample(question="q1", answer="a1", contexts=["c"]),
        GenerationSample(question="q2", answer="a2", contexts=["c"]),
    ]
    result = await evaluate_generation(samples, judge=judge)
    assert result.n_samples == 2
    assert result.avg_faithfulness == 0.9
    assert result.avg_relevancy == 0.9


@pytest.mark.asyncio
async def test_evaluate_empty():
    judge = FakeJudge("{}")
    result = await evaluate_generation([], judge=judge)
    assert result.n_samples == 0
    assert result.avg_faithfulness == 0.0

"""Tests cho LLM client (mock — không gọi real provider)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.llm_client import LLMClient, LLMMessage, LLMResponse
from src.generation.openai_compat_client import OpenAICompatClient


def test_llm_message_dataclass():
    m = LLMMessage(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_llm_response_defaults():
    r = LLMResponse(text="answer", model="m1")
    assert r.text == "answer"
    assert r.finish_reason == ""
    assert r.prompt_tokens == 0
    assert r.raw == {}


def test_openai_compat_construct_requires_fields():
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatClient(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError, match="api_key"):
        OpenAICompatClient(base_url="http://x/v1", api_key="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatClient(base_url="http://x/v1", api_key="k", model="")


def test_openai_compat_name_and_model():
    c = OpenAICompatClient(
        base_url="http://localhost:20128/v1",
        api_key="sk-test",
        model="cx/gpt-5.5",
    )
    assert "localhost:20128" in c.name
    assert c.model == "cx/gpt-5.5"
    assert isinstance(c, LLMClient)


@pytest.mark.asyncio
async def test_openai_compat_complete_mocked():
    """Mock AsyncOpenAI để test logic không cần real provider."""
    c = OpenAICompatClient(
        base_url="http://localhost:20128/v1",
        api_key="sk-test",
        model="cx/gpt-5.5",
    )

    mock_response = MagicMock()
    mock_response.id = "resp_1"
    mock_response.created = 1700000000
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Đây là câu trả lời."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20

    c._client.chat.completions.create = AsyncMock(return_value=mock_response)

    response = await c.complete(
        [LLMMessage(role="user", content="hỏi")],
        max_tokens=50,
        temperature=0.0,
    )
    assert response.text == "Đây là câu trả lời."
    assert response.model == "cx/gpt-5.5"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_openai_compat_stream_mocked():
    c = OpenAICompatClient(
        base_url="http://x/v1", api_key="k", model="m"
    )

    async def fake_stream():
        for piece in ["Hello", " ", "world", ".", ""]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = piece if piece else None
            yield chunk

    c._client.chat.completions.create = AsyncMock(return_value=fake_stream())

    pieces = []
    async for tok in c.complete_stream([LLMMessage(role="user", content="hi")]):
        pieces.append(tok)
    assert "".join(pieces) == "Hello world."


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_9router_smoke():
    """Real call to 9router (chỉ run khi user có 9router chạy + key trong .env).

    pytest -m integration để run.
    """
    from src.config import settings

    key = settings.custom_llm_api_key.get_secret_value()
    if settings.llm_provider != "custom" or not key:
        pytest.skip("LLM_PROVIDER != custom or no key — skip real test")

    c = OpenAICompatClient(
        base_url=settings.custom_llm_base_url,
        api_key=key,
        model=settings.custom_llm_model or settings.llm_model,
    )
    r = await c.complete(
        [LLMMessage(role="user", content="Trả lời 1 từ: hello")],
        max_tokens=20,
    )
    assert r.text
    assert r.model
    assert r.latency_ms > 0

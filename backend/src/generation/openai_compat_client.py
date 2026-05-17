"""LLM client cho mọi provider OpenAI-compatible.

Test với:
- OpenRouter (`https://openrouter.ai/api/v1`)
- 9router (`http://localhost:20128/v1`)
- DeepSeek direct (`https://api.deepseek.com/v1`)

Same OpenAI SDK + chỉ khác `base_url` + `api_key`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .llm_client import LLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatClient(LLMClient):
    """Async client cho OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        if not base_url:
            raise ValueError("base_url required")
        if not api_key:
            raise ValueError("api_key required")
        if not model:
            raise ValueError("model required")

        self._base_url = base_url
        self._model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return f"openai-compat({self._base_url})"

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.time()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = int((time.time() - t0) * 1000)

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        usage = response.usage

        return LLMResponse(
            text=text,
            model=self._model,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            raw={"id": response.id, "created": response.created},
        )

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

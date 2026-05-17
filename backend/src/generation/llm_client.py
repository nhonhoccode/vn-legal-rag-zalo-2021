"""Abstract LLM client + dataclasses.

Hỗ trợ 2 chế độ:
- Non-streaming: `complete()` → full LLMResponse.
- Streaming: `complete_stream()` → async iterator của text chunks.

Tất cả providers OpenAI-compatible (openrouter, 9router, deepseek) dùng cùng
`OpenAICompatClient` ở `openai_compat_client.py`. Gemini native sẽ implement riêng
nếu sau cần (chưa cần — user dùng 9router).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMMessage:
    """Một message trong conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Full response từ LLM."""

    text: str
    model: str
    finish_reason: str = ""  # "stop" | "length" | "content_filter" | ...
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)  # raw response provider để debug


class LLMClient(ABC):
    """Abstract async LLM client."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Async iterator yield text chunks."""
        # Subclass override; this stub satisfies type system.
        if False:
            yield ""  # pragma: no cover

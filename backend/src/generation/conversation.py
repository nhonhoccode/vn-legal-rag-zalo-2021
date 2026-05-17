"""Multi-turn conversation session manager.

Storage: Redis key `session:{id}` → JSON-encoded list of turns.

Mỗi turn: `{role: "user"|"assistant", content: str, timestamp: int, citations: list}`.

History truncation:
- Giữ last N turns (default 6 = 3 cặp user-assistant) để tránh prompt budget bùng nổ.
- Strict char limit (default 8000 chars) — drop oldest nếu vượt.

TTL: 1h default (config qua `redis_ttl_session`).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from src.cache.redis_client import AsyncRedisProtocol

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class Turn:
    """Một turn trong conversation."""

    role: Role
    content: str
    timestamp: int = 0
    citations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=int(d.get("timestamp", 0)),
            citations=d.get("citations") or [],
        )


class SessionManager:
    """CRUD session history trong Redis."""

    KEY_PREFIX = "session:"

    def __init__(
        self,
        redis: AsyncRedisProtocol,
        *,
        ttl_seconds: int = 3600,
        max_turns: int = 6,
        max_chars: int = 8000,
    ) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_chars = max_chars

    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    async def get_history(self, session_id: str) -> list[Turn]:
        """Trả về full history (đã truncate). Empty list nếu session chưa tồn tại."""
        raw = await self.redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Corrupt session %s: %s — return empty", session_id, e)
            return []
        if not isinstance(data, list):
            return []
        return [Turn.from_dict(t) for t in data if isinstance(t, dict)]

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        """Append 1 turn; tự truncate; refresh TTL."""
        if turn.timestamp == 0:
            turn.timestamp = int(time.time())
        history = await self.get_history(session_id)
        history.append(turn)
        history = self._truncate(history)
        await self._save(session_id, history)

    async def append_pair(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        citations: list[dict] | None = None,
    ) -> None:
        """Append user turn + assistant turn cùng lúc (common use case)."""
        history = await self.get_history(session_id)
        ts = int(time.time())
        history.append(Turn(role="user", content=user_query, timestamp=ts))
        history.append(
            Turn(
                role="assistant",
                content=assistant_answer,
                timestamp=ts,
                citations=citations or [],
            )
        )
        history = self._truncate(history)
        await self._save(session_id, history)

    async def clear(self, session_id: str) -> bool:
        deleted = await self.redis.delete(self._key(session_id))
        return deleted > 0

    async def exists(self, session_id: str) -> bool:
        return bool(await self.redis.exists(self._key(session_id)))

    async def _save(self, session_id: str, history: list[Turn]) -> None:
        payload = json.dumps([t.to_dict() for t in history], ensure_ascii=False)
        await self.redis.set(self._key(session_id), payload, ex=self.ttl_seconds)

    def _truncate(self, history: list[Turn]) -> list[Turn]:
        """Truncate by turn count + char budget. Drop oldest first."""
        # By turn count
        if len(history) > self.max_turns:
            history = history[-self.max_turns :]

        # By char budget
        total = sum(len(t.content) for t in history)
        while total > self.max_chars and len(history) > 2:
            dropped = history.pop(0)
            total -= len(dropped.content)

        return history

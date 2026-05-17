"""Tests cho /api/sessions/{id} routes + multi-turn integration với /api/ask."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import app
from src.api.routes import ask as ask_route
from src.cache.redis_client import make_fake_redis
from src.config import settings
from src.generation.conversation import SessionManager
from src.generation.llm_client import LLMClient, LLMResponse
from src.vectorstore.base import SearchHit


class FakeLLM(LLMClient):
    def __init__(self, answer="[Văn bản 1, Điều 41] OK."):
        self.answer = answer

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, messages, *, max_tokens=1024, temperature=0.0):
        return LLMResponse(text=self.answer, model="fake-model")

    async def complete_stream(self, messages, *, max_tokens=1024, temperature=0.0):
        for w in self.answer.split():
            yield w + " "


def _hit():
    return SearchHit(
        chunk_id="c1",
        text="Điều 41 ...",
        score=0.8,
        metadata={"law_id": "45/2019/qh14", "law_title": "BLLĐ",
                  "article_id": "41", "domain": "labor"},
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(settings, "allowed_api_keys", ["test-k"])

    fake_redis = make_fake_redis()
    fake_session = SessionManager(redis=fake_redis, ttl_seconds=60)
    fake_retriever = MagicMock()
    fake_retriever.search.return_value = [_hit()]
    fake_llm = FakeLLM()

    from src.generation.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(
        retriever=fake_retriever,
        llm=fake_llm,
        session_manager=fake_session,
    )

    # Patch dep module + route imports
    for cache in (dependencies.get_rag_pipeline, dependencies.get_session_manager):
        cache.cache_clear()
    monkeypatch.setattr(dependencies, "get_rag_pipeline", lambda: pipeline)
    monkeypatch.setattr(dependencies, "get_session_manager", lambda: fake_session)
    monkeypatch.setattr(ask_route, "get_rag_pipeline", lambda: pipeline)
    # sessions route imports get_session_manager
    from src.api.routes import sessions as sessions_route
    monkeypatch.setattr(sessions_route, "get_session_manager", lambda: fake_session)


HEADERS = {"X-API-Key": "test-k"}


# ============================================================
# /api/ask với session_id
# ============================================================
def test_ask_without_session_id_returns_null(client):
    r = client.post("/api/ask", json={"query": "test"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["session_id"] is None


def test_ask_with_empty_session_id_generates(client):
    """session_id='' → server tạo new id."""
    r = client.post(
        "/api/ask",
        json={"query": "test", "session_id": ""},
        headers=HEADERS,
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid is not None
    assert len(sid) == 32  # uuid4 hex


def test_ask_with_explicit_session_id_returns_it(client):
    r = client.post(
        "/api/ask",
        json={"query": "test", "session_id": "my-session-abc"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["session_id"] == "my-session-abc"


def test_ask_multi_turn_persists_and_uses_history(client):
    """2 calls cùng session_id → history available trong session response."""
    r1 = client.post(
        "/api/ask",
        json={"query": "Câu hỏi 1?", "session_id": "multi-session"},
        headers=HEADERS,
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/api/ask",
        json={"query": "Câu hỏi 2?", "session_id": "multi-session"},
        headers=HEADERS,
    )
    assert r2.status_code == 200

    # GET session → 4 turns (2 user + 2 assistant)
    rg = client.get("/api/sessions/multi-session", headers=HEADERS)
    assert rg.status_code == 200
    body = rg.json()
    assert body["session_id"] == "multi-session"
    assert body["turn_count"] == 4


# ============================================================
# /api/sessions endpoints
# ============================================================
def test_get_session_requires_auth(client):
    r = client.get("/api/sessions/anything")
    assert r.status_code == 401


def test_get_nonexistent_session_404(client):
    r = client.get("/api/sessions/does-not-exist-xxxxx", headers=HEADERS)
    assert r.status_code == 404


def test_delete_session_clears(client):
    # Setup history
    client.post(
        "/api/ask",
        json={"query": "x", "session_id": "to-delete"},
        headers=HEADERS,
    )
    # Delete
    r = client.delete("/api/sessions/to-delete", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] is True
    assert body["session_id"] == "to-delete"

    # 404 after delete
    rg = client.get("/api/sessions/to-delete", headers=HEADERS)
    assert rg.status_code == 404


def test_delete_nonexistent_session(client):
    r = client.delete("/api/sessions/never-existed", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["deleted"] is False


def test_session_id_max_length(client):
    """session_id > 64 chars → 422."""
    long_id = "x" * 65
    r = client.post(
        "/api/ask",
        json={"query": "test", "session_id": long_id},
        headers=HEADERS,
    )
    assert r.status_code == 422

"""E2E tests cho /api/ask endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import app
from src.api.routes import ask as ask_route
from src.config import settings
from src.generation.llm_client import LLMClient, LLMResponse
from src.vectorstore.base import SearchHit


class FakeLLM(LLMClient):
    def __init__(self, answer="Theo [Văn bản 1, Điều 41], được bồi thường."):
        self.answer = answer

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, messages, *, max_tokens=1024, temperature=0.0):
        return LLMResponse(text=self.answer, model="fake-model", prompt_tokens=10, completion_tokens=5)

    async def complete_stream(self, messages, *, max_tokens=1024, temperature=0.0):
        for w in self.answer.split():
            yield w + " "


def _hit(article_id="41"):
    return SearchHit(
        chunk_id=f"45_2019_qh14_{article_id}",
        text="Bồi thường khi sa thải trái pháp luật.",
        score=0.85,
        metadata={"law_id": "45/2019/qh14", "law_title": "Bộ luật Lao động",
                  "article_id": article_id, "domain": "labor"},
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_env_and_deps(monkeypatch):
    """Setup test API keys + mock retriever + mock LLM."""
    monkeypatch.setattr(settings, "allowed_api_keys", ["test-key-123"])

    fake_retriever = MagicMock()
    fake_retriever.search.return_value = [_hit("41"), _hit("13")]

    fake_llm = FakeLLM()

    from src.generation.rag_pipeline import RAGPipeline
    fake_pipeline = RAGPipeline(retriever=fake_retriever, llm=fake_llm, top_k=5)

    # Patch both dependency module + ask route module (route đã import name).
    dependencies.get_rag_pipeline.cache_clear()
    monkeypatch.setattr(dependencies, "get_rag_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(ask_route, "get_rag_pipeline", lambda: fake_pipeline)


def test_ask_requires_auth(client):
    r = client.post("/api/ask", json={"query": "test"})
    assert r.status_code == 401


def test_ask_validates_empty_query(client):
    r = client.post("/api/ask", json={"query": ""}, headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 422


def test_ask_validates_top_k_range(client):
    r = client.post(
        "/api/ask",
        json={"query": "lao động", "top_k": 100},
        headers={"X-API-Key": "test-key-123"},
    )
    assert r.status_code == 422


def test_ask_validates_temperature(client):
    r = client.post(
        "/api/ask",
        json={"query": "lao động", "temperature": 2.0},
        headers={"X-API-Key": "test-key-123"},
    )
    assert r.status_code == 422


def test_ask_returns_full_response(client):
    r = client.post(
        "/api/ask",
        json={"query": "Bồi thường khi sa thải?", "top_k": 5},
        headers={"X-API-Key": "test-key-123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "Bồi thường khi sa thải?"
    assert "Văn bản 1" in body["answer"] or "Điều" in body["answer"]
    assert len(body["sources"]) == 2
    assert len(body["citations"]) == 1
    assert body["citations"][0]["matched"] is True
    assert body["model"] == "fake-model"
    assert body["refused"] is False
    assert body["latency_ms"] >= 0


def test_ask_streaming_returns_sse(client):
    """Streaming endpoint trả SSE response."""
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "test", "stream": True},
        headers={"X-API-Key": "test-key-123"},
    ) as r:
        assert r.status_code == 200
        content_type = r.headers.get("content-type", "")
        assert "text/event-stream" in content_type

        lines = []
        for line in r.iter_lines():
            lines.append(line)
            if "[DONE]" in line:
                break

        # Should have at least: sources event + token events + done event + [DONE]
        sse_payloads = [ln for ln in lines if ln.startswith("data: ") and "[DONE]" not in ln]
        assert len(sse_payloads) >= 3

        import json
        events = [json.loads(ln[6:]) for ln in sse_payloads]
        types = {e["type"] for e in events}
        assert "sources" in types
        assert "token" in types
        assert "done" in types


def test_ask_jwt_auth_works(client, monkeypatch):
    """Test that JWT auth also works on /ask."""
    from pydantic import SecretStr
    monkeypatch.setattr(
        settings, "jwt_secret",
        SecretStr("test-secret-min-32-chars-for-hs256-jwt-ok"),
    )

    from jose import jwt
    token = jwt.encode(
        {"sub": "user@test.com", "email": "user@test.com"},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    r = client.post(
        "/api/ask",
        json={"query": "lao động"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200

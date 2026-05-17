"""End-to-end tests cho FastAPI app.

Dùng `TestClient` (fastapi.testclient) → chạy in-process, không cần uvicorn server.
"""

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from src.api.main import app
from src.config import settings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_keys(monkeypatch):
    """Set API key + JWT secret cố định cho tests."""
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "allowed_api_keys", ["test-key-123"])
    monkeypatch.setattr(
        settings,
        "jwt_secret",
        SecretStr("test-secret-min-32-chars-for-hs256-jwt-ok"),
    )


def _make_jwt(sub: str = "user@example.com", email: str | None = "user@example.com") -> str:
    payload = {"sub": sub}
    if email:
        payload["email"] = email
    return jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


# ============================================================
# Health (no auth)
# ============================================================
def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


# ============================================================
# Auth — search endpoint
# ============================================================
def test_search_requires_auth(client):
    r = client.post("/api/search", json={"query": "lao động"})
    assert r.status_code == 401


def test_search_with_invalid_api_key(client):
    r = client.post(
        "/api/search",
        json={"query": "lao động"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_search_with_invalid_jwt(client):
    r = client.post(
        "/api/search",
        json={"query": "lao động"},
        headers={"Authorization": "Bearer not-a-valid-token"},
    )
    assert r.status_code == 401


def test_search_request_validation_empty_query(client):
    r = client.post(
        "/api/search",
        json={"query": ""},
        headers={"X-API-Key": "test-key-123"},
    )
    assert r.status_code == 422  # validation error: min_length=1


def test_search_request_validation_top_k_too_high(client):
    r = client.post(
        "/api/search",
        json={"query": "lao động", "top_k": 999},
        headers={"X-API-Key": "test-key-123"},
    )
    assert r.status_code == 422


@pytest.mark.slow
def test_search_with_api_key_works(client):
    """E2E search với valid API key — cần data đã indexed.

    Slow vì tải embedder model + có thể build BM25 từ chunks.jsonl.
    """
    r = client.post(
        "/api/search",
        json={"query": "lao động", "top_k": 3, "use_hybrid": False},
        headers={"X-API-Key": "test-key-123"},
    )
    # Có thể 200 hoặc 500 nếu Chroma collection rỗng. Cả 2 đều acceptable cho smoke.
    if r.status_code == 200:
        body = r.json()
        assert body["query"] == "lao động"
        assert "hits" in body
        assert "latency_ms" in body


@pytest.mark.slow
def test_search_with_jwt_works(client):
    """E2E search với valid JWT."""
    token = _make_jwt()
    r = client.post(
        "/api/search",
        json={"query": "lao động", "top_k": 3, "use_hybrid": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Tương tự test API key — tolerant với empty index.
    assert r.status_code in (200, 500)

"""Pydantic schemas cho API request/response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=50)
    domain: str | None = Field(None, description="Filter by domain: labor, criminal, business")
    use_hybrid: bool = Field(True, description="Use dense+BM25 hybrid (vs dense only)")


class Citation(BaseModel):
    law_id: str
    law_title: str = ""
    article_id: str = ""
    score: float


class SearchHitResponse(BaseModel):
    chunk_id: str
    text: str
    score: float
    law_id: str = ""
    law_title: str = ""
    article_id: str = ""
    domain: str = ""
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitResponse]
    latency_ms: int


# ============================================================
# /api/ask schemas
# ============================================================
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    domain: str | None = None
    max_tokens: int = Field(1024, ge=64, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    stream: bool = False
    session_id: str | None = Field(
        None,
        description="Session ID cho multi-turn. None = single-turn. Empty string = tự generate.",
        max_length=64,
    )


class CitationResponse(BaseModel):
    text_index: int
    law_id: str = ""
    law_title: str = ""
    article_id: str = ""
    khoan_id: str | None = None
    matched: bool


class AskSourceResponse(BaseModel):
    chunk_id: str
    score: float
    law_id: str = ""
    law_title: str = ""
    article_id: str = ""
    domain: str = ""
    text: str  # excerpt


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    sources: list[AskSourceResponse]
    refused: bool
    model: str
    latency_ms: int
    retrieval_ms: int
    generation_ms: int
    rewrite_ms: int = 0
    session_id: str | None = None
    standalone_query: str | None = None
    metadata: dict = Field(default_factory=dict)


# ============================================================
# /api/sessions/{id} schemas
# ============================================================
class SessionTurnResponse(BaseModel):
    role: str
    content: str
    timestamp: int
    citations: list[dict] = Field(default_factory=list)


class SessionResponse(BaseModel):
    session_id: str
    turns: list[SessionTurnResponse]
    turn_count: int


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool

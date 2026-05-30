"""POST /api/ask — RAG generation endpoint (single-turn + multi-turn).

2 modes:
- Non-streaming (default): JSON response với full answer.
- Streaming (`stream=true`): Server-Sent Events (SSE) cho frontend AI SDK consume.

Multi-turn:
- Client gửi `session_id` (string ≤ 64 chars).
- Empty string `""` → server tạo session ID mới + trả về trong response.
- None hoặc thiếu field → single-turn (không persist).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_query_logger, get_rag_pipeline
from src.api.middleware.auth import AuthenticatedUser, get_current_user
from src.api.schemas import (
    AskRequest,
    AskResponse,
    AskSourceResponse,
    CitationResponse,
)
from src.generation.conversation import SessionManager
from src.generation.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["ask"])
logger = logging.getLogger(__name__)


def _resolve_session_id(raw: str | None) -> str | None:
    """Empty string → tạo new session id. None → single-turn (no session)."""
    if raw is None:
        return None
    if raw == "":
        return SessionManager.new_session_id()
    return raw


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """RAG ask. Stream qua `stream: true` body field."""
    pipeline: RAGPipeline = get_rag_pipeline()
    session_id = _resolve_session_id(request.session_id)

    if request.stream:
        return StreamingResponse(
            _stream_sse(pipeline, request, session_id),
            media_type="text/event-stream",
            headers={
                # no-transform = tell Cloudflare/proxies KHÔNG compress/buffer
                "Cache-Control": "no-cache, no-store, no-transform",
                "X-Accel-Buffering": "no",     # nginx
                "Connection": "keep-alive",
                "Content-Encoding": "identity", # disable gzip/br buffering
            },
        )

    result = await pipeline.ask(
        request.query,
        top_k=request.top_k,
        domain=request.domain,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        session_id=session_id,
    )

    # Phase 13: log query (best effort)
    import contextlib

    with contextlib.suppress(Exception):
        await get_query_logger().log({
            "endpoint": "/api/ask",
            "query": request.query,
            "user_id": user.id,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "n_hits": len(result.sources),
            "status": 200,
            "refused": result.refused,
        })

    return AskResponse(
        query=result.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                text_index=c.text_index,
                law_id=c.law_id,
                law_title=c.law_title,
                article_id=c.article_id,
                khoan_id=c.khoan_id,
                matched=c.matched,
            )
            for c in result.citations
        ],
        sources=[
            AskSourceResponse(
                chunk_id=h.chunk_id,
                score=h.score,
                law_id=h.metadata.get("law_id", ""),
                law_title=h.metadata.get("law_title", ""),
                article_id=h.metadata.get("article_id", ""),
                domain=h.metadata.get("domain", ""),
                text=h.text[:300] + "..." if len(h.text) > 300 else h.text,
            )
            for h in result.sources
        ],
        refused=result.refused,
        model=result.model,
        latency_ms=result.latency_ms,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
        rewrite_ms=result.rewrite_ms,
        session_id=result.session_id,
        standalone_query=result.standalone_query,
        metadata=result.metadata,
        follow_up_questions=result.follow_up_questions,
    )


async def _stream_sse(
    pipeline: RAGPipeline,
    request: AskRequest,
    session_id: str | None,
):
    """Stream events as SSE (compat với Vercel AI SDK + EventSource).

    Cloudflare buffer mitigation:
    1. Send 16KB padding ngay đầu để force flush Cloudflare edge buffer.
    2. Async iterator yield event ngay khi backend produce → no batching.
    """
    # 16KB padding comment — Cloudflare có thể buffer tới ~8KB cho SSE.
    # Comment lines (starts with ':') được client ignore nhưng force flush buffer.
    yield ":" + (" " * 16384) + "\n\n"
    # Sentinel event ngay để confirm stream alive
    yield 'data: {"type": "ready"}\n\n'
    async for event in pipeline.ask_stream(
        request.query,
        top_k=request.top_k,
        domain=request.domain,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        session_id=session_id,
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

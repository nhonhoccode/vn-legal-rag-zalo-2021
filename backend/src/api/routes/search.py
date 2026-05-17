"""Search route: `/api/search`."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from src.api.dependencies import get_dense_retriever, get_hybrid_retriever
from src.api.middleware.auth import AuthenticatedUser, get_current_user
from src.api.schemas import SearchHitResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> SearchResponse:
    t0 = time.time()

    where: dict | None = None
    if request.domain:
        where = {"domain": request.domain}

    if request.use_hybrid:
        retriever = get_hybrid_retriever()
        hits = retriever.search(
            request.query,
            top_k_final=request.top_k,
            where=where,
        )
    else:
        dense = get_dense_retriever()
        hits = dense.search(request.query, top_k=request.top_k, where=where)

    elapsed = int((time.time() - t0) * 1000)

    return SearchResponse(
        query=request.query,
        latency_ms=elapsed,
        hits=[
            SearchHitResponse(
                chunk_id=h.chunk_id,
                text=h.text,
                score=h.score,
                law_id=h.metadata.get("law_id", ""),
                law_title=h.metadata.get("law_title", ""),
                article_id=h.metadata.get("article_id", ""),
                domain=h.metadata.get("domain", ""),
                metadata={k: v for k, v in h.metadata.items() if k not in ("law_id", "law_title", "article_id", "domain")},
            )
            for h in hits
        ],
    )

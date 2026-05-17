"""FastAPI app entry point.

Run dev:
    cd project_AI/backend
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routes.ask import router as ask_router
from src.api.routes.search import router as search_router
from src.api.routes.sessions import router as sessions_router
from src.api.routes.stats import router as stats_router
from src.api.schemas import HealthResponse
from src.config import settings

# Suppress noisy chromadb telemetry warnings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Rate limit per IP (anonymous) hoặc per user.id (authed).
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])

app = FastAPI(
    title="VN Legal RAG API",
    version="0.1.0",
    description="Vietnamese legal RAG + Semantic Search backend.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """No-auth healthcheck."""
    return HealthResponse()


app.include_router(search_router)
app.include_router(ask_router)
app.include_router(sessions_router)
app.include_router(stats_router)

# Note: Phase 13 query logging gọi trực tiếp trong /api/ask route handler
# (KHÔNG dùng BaseHTTPMiddleware vì incompat với streaming SSE).

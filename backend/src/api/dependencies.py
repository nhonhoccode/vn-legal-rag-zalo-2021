"""FastAPI dependencies: build retriever singletons khi app startup."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from src.cache.redis_client import AsyncRedisProtocol, make_redis_client
from src.config import PROJECT_ROOT, settings
from src.embeddings.sentence_transformer_embedder import (
    STEmbedder,
    make_default_embedder,
    make_small_embedder,
)
from src.generation.conversation import SessionManager
from src.generation.llm_client import LLMClient
from src.generation.openai_compat_client import OpenAICompatClient
from src.generation.rag_pipeline import RAGPipeline
from src.reranking.bge_reranker import (
    CrossEncoderReranker,
    Reranker,
    make_default_reranker,
    make_small_reranker,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> STEmbedder:
    """Singleton embedder. Default = bge-m3 (Decision 3); fallback small nếu test."""
    if settings.embedding_model == "BAAI/bge-m3":
        return make_default_embedder(device=settings.embedding_device)
    if "MiniLM" in settings.embedding_model or "small" in settings.embedding_model.lower():
        return make_small_embedder(device=settings.embedding_device)
    return STEmbedder(model_name=settings.embedding_model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaStore:
    """Singleton ChromaDB."""
    persist = (
        Path(settings.chroma_persist_dir)
        if Path(settings.chroma_persist_dir).is_absolute()
        else PROJECT_ROOT / settings.chroma_persist_dir.lstrip("./")
    )
    return ChromaStore(persist_dir=persist, collection_name=settings.chroma_collection)


@lru_cache(maxsize=1)
def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(embedder=get_embedder(), store=get_vector_store())


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever | None:
    """Build BM25 từ chunks.jsonl nếu file tồn tại; cache pickle ở data/processed/bm25.pkl.

    Returns None nếu không có chunks → API sẽ chỉ dùng dense.
    """
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    bm25_pkl = PROJECT_ROOT / "data" / "processed" / "bm25.pkl"

    bm25 = BM25Retriever()
    if bm25_pkl.exists():
        try:
            bm25.load(bm25_pkl)
            logger.info("Loaded BM25 cache: %d docs", bm25.count())
            return bm25
        except Exception as e:  # noqa: BLE001
            logger.warning("BM25 cache load failed: %s — rebuild.", e)

    if not chunks_path.exists():
        logger.warning("No chunks.jsonl tại %s — BM25 disabled.", chunks_path)
        return None

    chunks: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        return None

    logger.info("Building BM25 từ %d chunks...", len(chunks))
    bm25.index(chunks)
    bm25.save(bm25_pkl)
    logger.info("BM25 saved → %s", bm25_pkl)
    return bm25


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """Lazy reranker singleton. None nếu disabled."""
    if not settings.enable_rerank:
        return None
    model = settings.reranker_model
    # Heuristic: small model nếu mention "MiniLM" or "small"
    if "MiniLM" in model or "small" in model.lower():
        return make_small_reranker(device=settings.embedding_device)
    if model == "BAAI/bge-reranker-v2-m3":
        return make_default_reranker(device=settings.embedding_device)
    return CrossEncoderReranker(model_name=model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense=get_dense_retriever(),
        bm25=get_bm25_retriever(),
        reranker=get_reranker(),
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Build LLM client từ settings (Decision 1 + custom 9router).

    Map LLM_PROVIDER → base URL:
      - openrouter → https://openrouter.ai/api/v1
      - deepseek   → https://api.deepseek.com/v1
      - custom     → settings.custom_llm_base_url (vd 9router localhost)
      - gemini     → NotImplementedError (chưa support — defer)
    """
    provider = settings.llm_provider
    model = settings.llm_model

    if provider == "openrouter":
        key = settings.openrouter_api_key.get_secret_value()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in .env")
        return OpenAICompatClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
        )
    if provider == "deepseek":
        key = settings.deepseek_api_key.get_secret_value()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
        return OpenAICompatClient(
            base_url="https://api.deepseek.com/v1",
            api_key=key,
            model=model,
        )
    if provider == "custom":
        key = settings.custom_llm_api_key.get_secret_value()
        if not settings.custom_llm_base_url:
            raise RuntimeError("CUSTOM_LLM_BASE_URL not set in .env")
        if not key:
            raise RuntimeError("CUSTOM_LLM_API_KEY not set in .env")
        return OpenAICompatClient(
            base_url=settings.custom_llm_base_url,
            api_key=key,
            model=settings.custom_llm_model or model,
        )
    if provider == "gemini":
        raise NotImplementedError(
            "Gemini native client chưa implement. Dùng `custom` provider qua proxy hoặc đổi llm_provider."
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@lru_cache(maxsize=1)
def get_redis_client() -> AsyncRedisProtocol:
    """Singleton Redis async client."""
    return make_redis_client(settings.redis_url)


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    """Singleton session manager (multi-turn history)."""
    return SessionManager(
        redis=get_redis_client(),
        ttl_seconds=settings.redis_ttl_session,
    )


@lru_cache(maxsize=1)
def get_query_logger():  # noqa: ANN201 — avoid import cycle
    """Singleton SQLite query logger (Phase 13)."""
    from src.api.middleware.query_log import QueryLogger

    return QueryLogger(db_path=PROJECT_ROOT / "data" / "app.db")


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retriever=get_hybrid_retriever(),
        llm=get_llm_client(),
        top_k=settings.top_k_final,
        session_manager=get_session_manager(),
    )

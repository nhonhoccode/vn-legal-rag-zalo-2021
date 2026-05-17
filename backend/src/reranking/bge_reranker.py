"""Cross-encoder reranker wrapper.

Default model: `BAAI/bge-reranker-v2-m3` (multilingual, mạnh tiếng Việt).
- ~1.1GB, lazy-load.
- Trên CPU: ~50-100ms cho 30 pairs.

Cho test/CI: dùng `make_small_reranker()` với MiniLM-based cross-encoder nhỏ hơn.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.vectorstore.base import SearchHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract reranker."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Sort hits theo cross-encoder score (descending). Trả về top-k nếu set."""


class CrossEncoderReranker(Reranker):
    """Sentence-transformers CrossEncoder wrapper."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        if not hits:
            return []

        model = self._load()
        pairs = [(query, h.text) for h in hits]
        scores = model.predict(pairs, show_progress_bar=False)

        # Replace score với rerank score, sort descending
        reranked: list[SearchHit] = []
        for h, s in zip(hits, scores, strict=True):
            reranked.append(
                SearchHit(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    score=float(s),
                    metadata={**h.metadata, "rerank_score": float(s), "original_score": h.score},
                )
            )
        reranked.sort(key=lambda h: h.score, reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked


def make_default_reranker(device: str = "cpu") -> CrossEncoderReranker:
    """Default = bge-reranker-v2-m3 (Decision 6)."""
    return CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device=device)


def make_small_reranker(device: str = "cpu") -> CrossEncoderReranker:
    """Small ~80MB cho test."""
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=device
    )

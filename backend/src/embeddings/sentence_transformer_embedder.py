"""SentenceTransformer-based embedder.

Default dùng `BAAI/bge-m3` (1024 dim, multilingual mạnh tiếng Việt — Decision 3).
Cho test có thể swap sang model nhỏ hơn (`paraphrase-MiniLM-L3-v2`, 17MB) qua arg.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class STEmbedder(Embedder):
    """Wrapper quanh `sentence_transformers.SentenceTransformer`.

    Lazy-load model khi cần để import module nhanh (model bge-m3 ~2.27GB!).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension() or 0

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        # Ensure float32 + 2-D
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors


# Convenience presets.
def make_default_embedder(device: str = "cpu") -> STEmbedder:
    """Default = bge-m3 (Decision 3)."""
    return STEmbedder(model_name="BAAI/bge-m3", device=device)


def make_small_embedder(device: str = "cpu") -> STEmbedder:
    """Small model ~17MB cho test/CI offline-friendly."""
    return STEmbedder(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", device=device)

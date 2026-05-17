"""Abstract base cho embedding model wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Bi-encoder embedding wrapper."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension của output vector."""

    @abstractmethod
    def encode(self, texts: list[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode list of texts → ndarray shape (N, dim).

        Vectors phải L2-normalized nếu Embedder claim cosine-friendly.
        """

    def encode_one(self, text: str) -> np.ndarray:
        """Encode 1 text → vector shape (dim,)."""
        return self.encode([text])[0]

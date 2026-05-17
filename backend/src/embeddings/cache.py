"""Disk cache wrapper cho query embeddings.

Use case: query trùng → tránh re-encode (CPU bge-m3 ~80ms vs Redis lookup ~1ms).
KHÔNG cache chunk embeddings (ngược: chunks index 1 lần vào ChromaDB).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from diskcache import Cache


class EmbeddingCache:
    """File-based cache cho `(model_name, query) → vector`."""

    def __init__(self, cache_dir: Path | str, model_name: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._cache = Cache(str(self.cache_dir))

    def _key(self, query: str) -> str:
        h = hashlib.sha256(f"{self.model_name}|{query}".encode()).hexdigest()
        return h[:32]

    def get(self, query: str) -> np.ndarray | None:
        raw = self._cache.get(self._key(query))
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32)

    def set(self, query: str, vector: np.ndarray, expire_seconds: int | None = None) -> None:
        v = np.asarray(vector, dtype=np.float32)
        self._cache.set(self._key(query), v.tobytes(), expire=expire_seconds)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._cache.close()

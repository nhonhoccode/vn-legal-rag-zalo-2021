"""Abstract base cho vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class SearchHit:
    """Result từ vector search."""

    chunk_id: str
    text: str
    score: float  # higher = more similar (cosine similarity nếu store dùng cosine)
    metadata: dict


class VectorStore(ABC):
    """Abstract vector store interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    def delete_collection(self) -> None: ...

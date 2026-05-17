"""Dense retrieval qua vector store."""

from __future__ import annotations

from collections.abc import Sequence

from src.embeddings.base import Embedder
from src.vectorstore.base import SearchHit, VectorStore


class DenseRetriever:
    """Dense retrieval: embed query → vector search."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        *,
        top_k: int = 30,
        where: dict | None = None,
    ) -> list[SearchHit]:
        q_vec = self.embedder.encode_one(query)
        return self.store.query(
            query_embedding=q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec),
            top_k=top_k,
            where=where,
        )

    def search_with_vector(
        self,
        q_vec: Sequence[float],
        *,
        top_k: int = 30,
        where: dict | None = None,
    ) -> list[SearchHit]:
        """Bypass embed step nếu caller đã có vector (e.g. cache hit)."""
        return self.store.query(query_embedding=list(q_vec), top_k=top_k, where=where)

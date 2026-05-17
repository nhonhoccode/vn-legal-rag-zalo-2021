"""Hybrid retrieval = Dense + BM25 + RRF + optional Reranking."""

from __future__ import annotations

from src.reranking.bge_reranker import Reranker
from src.retrieval.bm25 import BM25Retriever, reciprocal_rank_fusion
from src.retrieval.dense import DenseRetriever
from src.vectorstore.base import SearchHit


class HybridRetriever:
    """Orchestrate Dense + BM25 → RRF → optional Reranking."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        top_k_final: int = 10,
        top_k_per_retriever: int = 30,
        where: dict | None = None,
        enable_rerank: bool | None = None,
    ) -> list[SearchHit]:
        """Pipeline: Dense + BM25 → RRF top_k_per_retriever → (optional rerank) → top_k_final.

        Args:
            enable_rerank: override class-level setting. None = dùng self.reranker
                          (rerank nếu reranker != None).
        """
        dense_hits = self.dense.search(query, top_k=top_k_per_retriever, where=where)

        # Fusion stage
        if self.bm25 is None:
            fused = dense_hits
        else:
            bm25_hits = self.bm25.search(query, top_k=top_k_per_retriever)
            if where:
                bm25_hits = [
                    h
                    for h in bm25_hits
                    if all(h.metadata.get(k) == v for k, v in where.items())
                ]
            # RRF lấy top_k_per_retriever để có pool đủ lớn cho reranker.
            fused = reciprocal_rank_fusion(
                [dense_hits, bm25_hits],
                top_k=top_k_per_retriever,
            )

        # Reranking stage (optional)
        use_rerank = enable_rerank if enable_rerank is not None else (self.reranker is not None)
        if use_rerank and self.reranker and fused:
            fused = self.reranker.rerank(query, fused, top_k=top_k_final)
        else:
            fused = fused[:top_k_final]

        return fused

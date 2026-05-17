"""Custom retrieval metrics: Recall@k, Precision@k, MRR, nDCG.

Input: golden Q&A dataset format giống Phase 2 GoldenQA:
    {question_id, question, relevant_articles: [{law_id, article_id}]}

Retrieval output: list[SearchHit] với metadata {law_id, article_id}.

Match: 1 chunk = relevant nếu (law_id, article_id) trùng với 1 entry trong golden.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.vectorstore.base import SearchHit


@dataclass(slots=True)
class GoldenItem:
    """Lightweight version cho eval (không phụ thuộc ingestion module)."""

    question_id: str
    question: str
    relevant_keys: set[tuple[str, str]]  # set of (law_id, article_id) cho fast lookup

    @classmethod
    def from_dict(cls, d: dict) -> GoldenItem:
        relevant_articles = d.get("relevant_articles") or []
        keys = {
            (str(a.get("law_id", "")), str(a.get("article_id", "")))
            for a in relevant_articles
        }
        return cls(
            question_id=str(d.get("question_id", "")),
            question=d.get("question", ""),
            relevant_keys=keys,
        )


def _hit_key(hit: SearchHit) -> tuple[str, str]:
    return (
        str(hit.metadata.get("law_id", "")),
        str(hit.metadata.get("article_id", "")),
    )


# ============================================================
# Per-query metrics
# ============================================================
def recall_at_k(hits: list[SearchHit], relevant_keys: set[tuple[str, str]], k: int) -> float:
    """Recall@k = 1 nếu ít nhất 1 relevant trong top-k, else 0.

    Convention: binary relevance (vẫn = 1 nếu nhiều relevant, vì golden thường có 1-2).
    """
    if not relevant_keys:
        return 0.0
    top_k = hits[:k]
    return float(any(_hit_key(h) in relevant_keys for h in top_k))


def precision_at_k(hits: list[SearchHit], relevant_keys: set[tuple[str, str]], k: int) -> float:
    if not relevant_keys or k <= 0:
        return 0.0
    top_k = hits[:k]
    if not top_k:
        return 0.0
    matched = sum(1 for h in top_k if _hit_key(h) in relevant_keys)
    return matched / k


def reciprocal_rank(hits: list[SearchHit], relevant_keys: set[tuple[str, str]]) -> float:
    """RR = 1 / rank của relevant đầu tiên (1-based). 0 nếu không match."""
    if not relevant_keys:
        return 0.0
    for rank, h in enumerate(hits, 1):
        if _hit_key(h) in relevant_keys:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[SearchHit], relevant_keys: set[tuple[str, str]], k: int) -> float:
    """nDCG@k với binary relevance. DCG@k / IDCG@k.

    IDCG cho binary với r relevant docs: sum_{i=1..min(r,k)} 1/log2(i+1).
    """
    if not relevant_keys or k <= 0:
        return 0.0
    top_k = hits[:k]
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, h in enumerate(top_k, 1)
        if _hit_key(h) in relevant_keys
    )
    n_rel = min(len(relevant_keys), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def f_beta_at_k(
    hits: list[SearchHit],
    relevant_keys: set[tuple[str, str]],
    k: int,
    *,
    beta: float = 2.0,
) -> float:
    """F-beta@k. Default β=2 → metric của Zalo AI Challenge 2021 Legal Retrieval.

    F_β = (1 + β²) · P · R / (β² · P + R). β=2 → recall-weighted.
    """
    if not relevant_keys or k <= 0:
        return 0.0
    p = precision_at_k(hits, relevant_keys, k)
    matched = sum(1 for h in hits[:k] if _hit_key(h) in relevant_keys)
    r = matched / len(relevant_keys)
    if p == 0.0 and r == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * p * r / (beta_sq * p + r)


# ============================================================
# Aggregated over dataset
# ============================================================
@dataclass(slots=True)
class RetrievalEvalResult:
    """Aggregated metrics over a golden dataset."""

    n_queries: int
    recall_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at: dict[int, float] = field(default_factory=dict)
    f2_at: dict[int, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "recall_at": self.recall_at,
            "precision_at": self.precision_at,
            "mrr": self.mrr,
            "ndcg_at": self.ndcg_at,
            "f2_at": self.f2_at,
        }


def evaluate_retrieval(
    golden: Iterable[GoldenItem],
    search_fn,  # Callable[[str], list[SearchHit]]
    *,
    k_values: list[int] = (1, 3, 5, 10),
) -> RetrievalEvalResult:
    """Run eval. Tính metrics per query, sau đó average."""
    golden_list = list(golden)
    n = len(golden_list)
    if n == 0:
        return RetrievalEvalResult(n_queries=0)

    recall_sums = {k: 0.0 for k in k_values}
    precision_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    f2_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    per_query: list[dict] = []

    for item in golden_list:
        hits = search_fn(item.question)
        rr = reciprocal_rank(hits, item.relevant_keys)
        rr_sum += rr

        record: dict = {
            "question_id": item.question_id,
            "rr": rr,
            "n_relevant": len(item.relevant_keys),
            "n_retrieved": len(hits),
        }
        for k in k_values:
            r = recall_at_k(hits, item.relevant_keys, k)
            p = precision_at_k(hits, item.relevant_keys, k)
            ndcg = ndcg_at_k(hits, item.relevant_keys, k)
            f2 = f_beta_at_k(hits, item.relevant_keys, k, beta=2.0)
            recall_sums[k] += r
            precision_sums[k] += p
            ndcg_sums[k] += ndcg
            f2_sums[k] += f2
            record[f"recall@{k}"] = r
            record[f"precision@{k}"] = p
            record[f"ndcg@{k}"] = ndcg
            record[f"f2@{k}"] = f2
        per_query.append(record)

    return RetrievalEvalResult(
        n_queries=n,
        recall_at={k: v / n for k, v in recall_sums.items()},
        precision_at={k: v / n for k, v in precision_sums.items()},
        mrr=rr_sum / n,
        ndcg_at={k: v / n for k, v in ndcg_sums.items()},
        f2_at={k: v / n for k, v in f2_sums.items()},
        per_query=per_query,
    )

"""Tests cho retrieval metrics."""

from src.eval.retrieval_metrics import (
    GoldenItem,
    evaluate_retrieval,
    f_beta_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.vectorstore.base import SearchHit


def _hit(law_id, article_id, score=0.5):
    return SearchHit(
        chunk_id=f"{law_id}_{article_id}",
        text="",
        score=score,
        metadata={"law_id": law_id, "article_id": article_id},
    )


def test_recall_at_k_hit_in_top_k():
    hits = [_hit("L1", "1"), _hit("L1", "2"), _hit("L2", "3")]
    rel = {("L1", "2")}
    assert recall_at_k(hits, rel, k=2) == 1.0
    assert recall_at_k(hits, rel, k=1) == 0.0  # L1/1 not relevant


def test_recall_at_k_no_relevant():
    hits = [_hit("L1", "1")]
    assert recall_at_k(hits, set(), k=5) == 0.0


def test_precision_at_k():
    hits = [_hit("L1", "1"), _hit("L1", "2"), _hit("L2", "3")]
    rel = {("L1", "1"), ("L2", "3")}
    # In top 3: 2 relevant → precision = 2/3
    assert abs(precision_at_k(hits, rel, k=3) - 2 / 3) < 1e-9


def test_reciprocal_rank():
    hits = [_hit("L1", "1"), _hit("L1", "2"), _hit("L1", "3")]
    rel = {("L1", "3")}
    # rank 3 → RR = 1/3
    assert abs(reciprocal_rank(hits, rel) - 1 / 3) < 1e-9


def test_reciprocal_rank_first_position():
    hits = [_hit("L1", "1")]
    rel = {("L1", "1")}
    assert reciprocal_rank(hits, rel) == 1.0


def test_reciprocal_rank_no_match():
    hits = [_hit("L1", "1")]
    rel = {("L2", "99")}
    assert reciprocal_rank(hits, rel) == 0.0


def test_ndcg_at_k_perfect():
    hits = [_hit("L1", "1"), _hit("L1", "2")]
    rel = {("L1", "1"), ("L1", "2")}
    # Both in top 2 + correct order → nDCG = 1.0
    assert abs(ndcg_at_k(hits, rel, k=2) - 1.0) < 1e-9


def test_ndcg_at_k_partial():
    hits = [_hit("L9", "0"), _hit("L1", "1")]  # relevant ở rank 2
    rel = {("L1", "1")}
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    import math
    expected = (1 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k(hits, rel, k=2) - expected) < 1e-9


def test_f2_at_k_hit_rank_1_single_relevant():
    """Zalo-style: 1 relevant article hit at rank 1, k=10.

    P = 1/10 = 0.1; R = 1/1 = 1.0
    F2 = 5 * 0.1 * 1.0 / (4 * 0.1 + 1.0) = 0.5 / 1.4 ≈ 0.3571
    """
    hits = [_hit("L1", "1")] + [_hit("L9", str(i)) for i in range(9)]
    rel = {("L1", "1")}
    expected = 5 * 0.1 * 1.0 / (4 * 0.1 + 1.0)
    assert abs(f_beta_at_k(hits, rel, k=10, beta=2.0) - expected) < 1e-9


def test_f2_at_k_perfect_recall_perfect_precision():
    """Perfect retrieval: 1 relevant hit at rank 1, k=1 → P=R=1, F2=1."""
    hits = [_hit("L1", "1")]
    rel = {("L1", "1")}
    assert abs(f_beta_at_k(hits, rel, k=1, beta=2.0) - 1.0) < 1e-9


def test_f2_at_k_no_hit():
    """No relevant in top-k → F2 = 0."""
    hits = [_hit("L9", "0")] * 5
    rel = {("L1", "1")}
    assert f_beta_at_k(hits, rel, k=5, beta=2.0) == 0.0


def test_f2_at_k_empty_relevant():
    """No ground truth → F2 = 0."""
    hits = [_hit("L1", "1")]
    assert f_beta_at_k(hits, set(), k=5, beta=2.0) == 0.0


def test_evaluate_retrieval_aggregates():
    golden = [
        GoldenItem(question_id="q1", question="?", relevant_keys={("L1", "1")}),
        GoldenItem(question_id="q2", question="?", relevant_keys={("L2", "5")}),
    ]

    def search(q):
        if q == "?":
            return [_hit("L1", "1"), _hit("L2", "5")]
        return []

    # Note: both queries use "?" so both return same hits.
    # q1 relevant L1/1 → at rank 1 → R@1=1, RR=1
    # q2 relevant L2/5 → at rank 2 → R@1=0, R@3=1, RR=0.5
    result = evaluate_retrieval(golden, search, k_values=[1, 3])
    assert result.n_queries == 2
    assert result.recall_at[1] == 0.5  # only q1 hit
    assert result.recall_at[3] == 1.0
    assert result.mrr == 0.75  # (1 + 0.5)/2
    # F2@3: q1 (1 rel @ rank 1, k=3) → P=1/3, R=1 → F2 = 5*(1/3)*1 / (4*(1/3)+1) = (5/3)/(7/3) = 5/7
    # F2@3: q2 (1 rel @ rank 2, k=3) → P=1/3, R=1 → 5/7 same
    assert 1 in result.f2_at and 3 in result.f2_at


def test_golden_item_from_dict():
    d = {
        "question_id": "q1",
        "question": "test",
        "relevant_articles": [{"law_id": "L1", "article_id": "1"}, {"law_id": "L2", "article_id": "5"}],
    }
    item = GoldenItem.from_dict(d)
    assert item.question_id == "q1"
    assert item.relevant_keys == {("L1", "1"), ("L2", "5")}


def test_empty_golden_returns_zero():
    result = evaluate_retrieval([], lambda q: [])
    assert result.n_queries == 0

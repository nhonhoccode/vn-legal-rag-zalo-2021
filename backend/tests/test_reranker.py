"""Tests cho Reranker + HybridRetriever integration."""

from unittest.mock import MagicMock

import pytest

from src.reranking.bge_reranker import (
    CrossEncoderReranker,
    Reranker,
    make_default_reranker,
    make_small_reranker,
)
from src.retrieval.hybrid import HybridRetriever
from src.vectorstore.base import SearchHit


def _hit(chunk_id="c", text="text", score=0.5, **meta):
    return SearchHit(chunk_id=chunk_id, text=text, score=score, metadata=meta)


# ============================================================
# Reranker construct + lazy load
# ============================================================
def test_make_default_reranker_constructs():
    r = make_default_reranker()
    assert isinstance(r, Reranker)
    assert r.name == "BAAI/bge-reranker-v2-m3"
    # Lazy: model not loaded yet
    assert r._model is None


def test_make_small_reranker_constructs():
    r = make_small_reranker()
    assert "MiniLM" in r.name
    assert r._model is None


def test_cross_encoder_reranker_custom_model():
    r = CrossEncoderReranker(model_name="my/custom-model")
    assert r.name == "my/custom-model"


def test_reranker_empty_hits():
    r = CrossEncoderReranker()
    assert r.rerank("query", []) == []


# ============================================================
# Reranker logic mocked (no real model)
# ============================================================
def test_reranker_sorts_by_score_descending():
    r = CrossEncoderReranker()
    # Mock predict to return reversed scores
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.2, 0.9, 0.5]
    r._model = fake_model

    hits = [
        _hit(chunk_id="a", text="A"),
        _hit(chunk_id="b", text="B"),
        _hit(chunk_id="c", text="C"),
    ]
    result = r.rerank("query", hits)
    # Order: b (0.9), c (0.5), a (0.2)
    assert [h.chunk_id for h in result] == ["b", "c", "a"]
    assert result[0].score == 0.9


def test_reranker_top_k():
    r = CrossEncoderReranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    r._model = fake_model

    hits = [_hit(chunk_id=str(i), text=str(i)) for i in range(5)]
    result = r.rerank("q", hits, top_k=2)
    assert len(result) == 2
    assert [h.chunk_id for h in result] == ["4", "3"]  # top 2 by score


def test_reranker_preserves_original_score_in_metadata():
    r = CrossEncoderReranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.95]
    r._model = fake_model

    hit = _hit(chunk_id="a", text="t", score=0.3, domain="labor")
    result = r.rerank("q", [hit])
    assert result[0].score == 0.95
    assert result[0].metadata["rerank_score"] == 0.95
    assert result[0].metadata["original_score"] == 0.3
    assert result[0].metadata["domain"] == "labor"  # original preserved


# ============================================================
# HybridRetriever + Reranker
# ============================================================
def _make_dense(hits):
    m = MagicMock()
    m.search.return_value = hits
    return m


class FakeReranker(Reranker):
    """Reranker đảo ngược thứ tự (cho test verify reranker được gọi)."""

    def __init__(self):
        self.called = False

    @property
    def name(self) -> str:
        return "fake-reverse"

    def rerank(self, query, hits, *, top_k=None):
        self.called = True
        reversed_hits = list(reversed(hits))
        if top_k:
            return reversed_hits[:top_k]
        return reversed_hits


def test_hybrid_uses_reranker_when_provided():
    dense_hits = [_hit(chunk_id=str(i)) for i in range(5)]
    dense = _make_dense(dense_hits)
    reranker = FakeReranker()

    hybrid = HybridRetriever(dense=dense, bm25=None, reranker=reranker)
    result = hybrid.search("query", top_k_final=3)

    assert reranker.called is True
    # FakeReranker reverses order; dense returned [0,1,2,3,4] → reversed [4,3,2,1,0] → top 3 = [4,3,2]
    assert [h.chunk_id for h in result] == ["4", "3", "2"]


def test_hybrid_skips_reranker_when_disabled():
    dense_hits = [_hit(chunk_id=str(i)) for i in range(5)]
    dense = _make_dense(dense_hits)
    reranker = FakeReranker()

    hybrid = HybridRetriever(dense=dense, bm25=None, reranker=reranker)
    result = hybrid.search("query", top_k_final=3, enable_rerank=False)

    assert reranker.called is False
    assert [h.chunk_id for h in result] == ["0", "1", "2"]


def test_hybrid_no_reranker_returns_truncated():
    dense_hits = [_hit(chunk_id=str(i)) for i in range(10)]
    dense = _make_dense(dense_hits)

    hybrid = HybridRetriever(dense=dense, bm25=None, reranker=None)
    result = hybrid.search("query", top_k_final=3)
    assert len(result) == 3


# ============================================================
# Slow: real cross-encoder
# ============================================================
@pytest.mark.slow
def test_real_small_reranker_predicts_scores():
    r = make_small_reranker()
    hits = [
        _hit(chunk_id="relevant", text="The cat sat on the mat."),
        _hit(chunk_id="irrelevant", text="Quantum mechanics is hard."),
    ]
    result = r.rerank("Where is the cat?", hits)
    assert len(result) == 2
    # "Where is the cat?" matches "cat sat on mat" better than quantum mechanics
    assert result[0].chunk_id == "relevant"
    assert result[0].score > result[1].score

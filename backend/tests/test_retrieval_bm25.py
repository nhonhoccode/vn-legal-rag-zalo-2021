"""Tests cho BM25Retriever + RRF."""

from src.retrieval.bm25 import (
    BM25Retriever,
    reciprocal_rank_fusion,
    tokenize_vi,
)
from src.vectorstore.base import SearchHit


def test_tokenize_vi_basic():
    tokens = tokenize_vi("Người lao động bị sa thải")
    assert len(tokens) > 0
    # pyvi join compound words với underscore
    joined = " ".join(tokens)
    assert "người_lao_động" in joined or "lao_động" in joined or "người" in tokens


def test_tokenize_vi_lowercase():
    tokens = tokenize_vi("Bộ Luật Lao Động")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_vi_strips_punctuation():
    tokens = tokenize_vi("Điều 41! Bồi thường?")
    # No punct in tokens
    for t in tokens:
        assert all(c not in t for c in "!?,;")


def test_tokenize_vi_empty():
    assert tokenize_vi("") == []
    assert tokenize_vi("   ") == []


def test_bm25_index_count():
    bm25 = BM25Retriever()
    assert bm25.count() == 0
    bm25.index([
        {"chunk_id": "a", "text": "Người lao động bị sa thải.", "law_id": "L1", "metadata": {}},
        {"chunk_id": "b", "text": "Tội trộm cắp tài sản.", "law_id": "L2", "metadata": {}},
    ])
    assert bm25.count() == 2


def test_bm25_search_returns_relevant():
    bm25 = BM25Retriever()
    bm25.index([
        {"chunk_id": "labor", "text": "Người lao động bị sa thải trái pháp luật được bồi thường.", "law_id": "L1", "domain": "labor", "metadata": {}},
        {"chunk_id": "criminal", "text": "Tội trộm cắp tài sản bị phạt tù.", "law_id": "L2", "domain": "criminal", "metadata": {}},
        {"chunk_id": "business", "text": "Doanh nghiệp đăng ký kinh doanh.", "law_id": "L3", "domain": "business", "metadata": {}},
    ])
    hits = bm25.search("lao động sa thải", top_k=3)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "labor"


def test_bm25_search_empty_query():
    bm25 = BM25Retriever()
    bm25.index([{"chunk_id": "a", "text": "test", "metadata": {}}])
    hits = bm25.search("", top_k=10)
    assert hits == []


def test_bm25_no_index_returns_empty():
    bm25 = BM25Retriever()
    assert bm25.search("query", top_k=10) == []


def test_bm25_save_load_round_trip(tmp_path):
    bm25 = BM25Retriever()
    bm25.index([
        {"chunk_id": "a", "text": "Người lao động bị sa thải.", "law_id": "L1", "metadata": {}},
        {"chunk_id": "b", "text": "Tội trộm cắp tài sản.", "law_id": "L2", "metadata": {}},
        {"chunk_id": "c", "text": "Doanh nghiệp đăng ký.", "law_id": "L3", "metadata": {}},
    ])
    pkl = tmp_path / "bm25.pkl"
    bm25.save(pkl)
    assert pkl.exists()

    bm25_2 = BM25Retriever()
    bm25_2.load(pkl)
    assert bm25_2.count() == 3
    hits = bm25_2.search("lao động sa thải", top_k=2)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "a"


def test_bm25_returns_searchhit():
    bm25 = BM25Retriever()
    bm25.index([
        {"chunk_id": "a", "text": "lao động hợp đồng tiền lương", "law_id": "L1", "metadata": {"x": 1}},
        {"chunk_id": "b", "text": "tội phạm hình sự", "law_id": "L2", "metadata": {}},
        {"chunk_id": "c", "text": "doanh nghiệp đầu tư", "law_id": "L3", "metadata": {}},
    ])
    hits = bm25.search("lao động", top_k=1)
    assert len(hits) >= 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].chunk_id == "a"


# ============================================================
# RRF
# ============================================================
def test_rrf_merges_two_rankings():
    h1 = [
        SearchHit(chunk_id="a", text="", score=0.9, metadata={}),
        SearchHit(chunk_id="b", text="", score=0.8, metadata={}),
    ]
    h2 = [
        SearchHit(chunk_id="b", text="", score=10.0, metadata={}),  # b ở rank 0 trong h2
        SearchHit(chunk_id="c", text="", score=8.0, metadata={}),
    ]
    fused = reciprocal_rank_fusion([h1, h2], top_k=3)
    ids = [h.chunk_id for h in fused]
    # b appear ở 2 lists → high RRF score → rank 1.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_top_k_limits():
    h1 = [SearchHit(chunk_id=f"id{i}", text="", score=1.0 - i*0.1, metadata={}) for i in range(10)]
    fused = reciprocal_rank_fusion([h1], top_k=3)
    assert len(fused) == 3


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([], top_k=5) == []
    assert reciprocal_rank_fusion([[]], top_k=5) == []


def test_rrf_preserves_metadata():
    h1 = [SearchHit(chunk_id="a", text="text-a", score=0.9, metadata={"key": "value"})]
    fused = reciprocal_rank_fusion([h1], top_k=1)
    assert fused[0].text == "text-a"
    assert fused[0].metadata == {"key": "value"}

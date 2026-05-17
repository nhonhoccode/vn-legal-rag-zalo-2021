"""Tests cho ChromaStore wrapper."""

import pytest

from src.vectorstore.base import SearchHit, VectorStore
from src.vectorstore.chroma_store import ChromaStore, _sanitize_metadata


def test_sanitize_metadata_preserves_primitives():
    md = {"a": "string", "b": 1, "c": 1.5, "d": True}
    out = _sanitize_metadata(md)
    assert out == md


def test_sanitize_metadata_drops_none():
    """ChromaDB reject None values → wrapper drop key."""
    md = {"a": "ok", "b": None, "c": 1}
    out = _sanitize_metadata(md)
    assert "b" not in out
    assert out["a"] == "ok"
    assert out["c"] == 1


def test_sanitize_metadata_converts_complex_to_str():
    md = {"list_field": [1, 2, 3], "dict_field": {"nested": True}}
    out = _sanitize_metadata(md)
    assert isinstance(out["list_field"], str)
    assert isinstance(out["dict_field"], str)


def test_chroma_store_lazy_init(tmp_path):
    """Construct không create collection ngay."""
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    assert store._client is None
    assert store._collection is None
    assert isinstance(store, VectorStore)


def test_chroma_store_count_empty(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    assert store.count() == 0


def test_chroma_store_add_count_query(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection_name="test")
    # Manual vectors (4-dim) cho test fast.
    store.add(
        ids=["a", "b", "c"],
        documents=["text A", "text B", "text C"],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # gần A
        ],
        metadatas=[
            {"domain": "labor"},
            {"domain": "criminal"},
            {"domain": "labor"},
        ],
    )
    assert store.count() == 3

    # Query gần [1,0,0,0] → expect a, c, b
    hits = store.query(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "a"
    assert hits[1].chunk_id == "c"
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].text == "text A"


def test_chroma_store_query_with_filter(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma2", collection_name="test")
    store.add(
        ids=["a", "b", "c"],
        documents=["x", "y", "z"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        metadatas=[
            {"domain": "labor"},
            {"domain": "criminal"},
            {"domain": "labor"},
        ],
    )
    hits = store.query(
        query_embedding=[1.0, 0.0],
        top_k=10,
        where={"domain": "labor"},
    )
    assert len(hits) == 2
    chunk_ids = {h.chunk_id for h in hits}
    assert chunk_ids == {"a", "c"}


def test_chroma_store_returns_searchhit_dataclass(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma3")
    store.add(
        ids=["a"],
        documents=["text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"meta": "value"}],
    )
    hits = store.query(query_embedding=[1.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].chunk_id == "a"
    assert hits[0].text == "text"
    assert hits[0].metadata == {"meta": "value"}


def test_chroma_store_persists_to_disk(tmp_path):
    """Tạo store, add, close. Tạo store mới ở same path → data còn."""
    persist = tmp_path / "chroma_persist"

    store1 = ChromaStore(persist_dir=persist, collection_name="test_persist")
    store1.add(
        ids=["a"],
        documents=["doc"],
        embeddings=[[0.5, 0.5]],
        metadatas=[{"k": "v"}],
    )
    assert store1.count() == 1

    # New instance — should see persisted data.
    store2 = ChromaStore(persist_dir=persist, collection_name="test_persist")
    assert store2.count() == 1


def test_chroma_store_delete_collection(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma4", collection_name="test_delete")
    store.add(
        ids=["a"], documents=["d"], embeddings=[[1.0, 0.0]], metadatas=[{"k": "v"}]
    )
    assert store.count() == 1
    store.delete_collection()
    # After delete, calling count() lại sẽ create new empty collection.
    assert store.count() == 0


def test_chroma_store_handles_empty_metadata(tmp_path):
    """Empty metadata {} → wrapper thêm placeholder để ChromaDB chấp nhận."""
    store = ChromaStore(persist_dir=tmp_path / "chroma_empty", collection_name="test_empty")
    store.add(
        ids=["a"], documents=["d"], embeddings=[[1.0, 0.0]], metadatas=[{}]
    )
    assert store.count() == 1
    hits = store.query(query_embedding=[1.0, 0.0], top_k=1)
    assert hits[0].metadata.get("_placeholder") == ""


def test_chroma_store_handles_complex_metadata(tmp_path):
    """Metadata có nested list/dict — phải convert sang str (không crash)."""
    store = ChromaStore(persist_dir=tmp_path / "chroma5")
    store.add(
        ids=["a"],
        documents=["d"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"complex": [1, 2, 3], "ok": "fine"}],
    )
    hits = store.query(query_embedding=[1.0, 0.0], top_k=1)
    assert hits[0].metadata.get("ok") == "fine"
    assert hits[0].metadata.get("complex") == "[1, 2, 3]"


def test_chroma_store_add_empty_noop(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma6")
    store.add(ids=[], documents=[], embeddings=[], metadatas=[])
    assert store.count() == 0


# ============================================================
# Integration with embedder (slow, real model)
# ============================================================
@pytest.mark.slow
def test_e2e_embed_then_query(tmp_path):
    """Smoke test full pipeline: embed sample chunks → query → verify top-1."""
    from src.embeddings.sentence_transformer_embedder import make_small_embedder

    embedder = make_small_embedder()
    store = ChromaStore(persist_dir=tmp_path / "e2e", collection_name="e2e_test")

    texts = [
        "Người lao động bị sa thải trái pháp luật được bồi thường ít nhất 2 tháng tiền lương.",
        "Tội trộm cắp tài sản có thể bị phạt tù.",
        "Doanh nghiệp phải đăng ký kinh doanh trước khi hoạt động.",
        "Thủ tục xuất nhập khẩu hàng hoá thực hiện theo quy định hải quan.",
    ]
    ids = [f"c{i}" for i in range(len(texts))]
    metas = [{"domain": d} for d in ["labor", "criminal", "business", "trade"]]
    vectors = embedder.encode(texts).tolist()

    store.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    assert store.count() == len(texts)

    # Query liên quan đến lao động → expect top-1 = labor chunk
    q_vec = embedder.encode_one("bồi thường khi bị sa thải").tolist()
    hits = store.query(query_embedding=q_vec, top_k=2)
    assert len(hits) == 2
    assert hits[0].metadata["domain"] == "labor"

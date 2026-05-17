"""Tests cho embeddings module.

Dùng `paraphrase-MiniLM-L3-v2` (~17MB) thay bge-m3 (2.27GB) cho test nhanh.
Real bge-m3 chỉ chạy trên Kaggle GPU notebook (xem `notebooks/kaggle_embed.ipynb`).
"""

import numpy as np
import pytest

from src.embeddings.base import Embedder
from src.embeddings.cache import EmbeddingCache
from src.embeddings.sentence_transformer_embedder import (
    STEmbedder,
    make_default_embedder,
    make_small_embedder,
)


# ============================================================
# Cache (no model load needed)
# ============================================================
def test_cache_set_get_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "ec", model_name="test-model")
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cache.get("hello") is None
    cache.set("hello", vec)
    got = cache.get("hello")
    np.testing.assert_array_equal(got, vec)
    cache.close()


def test_cache_different_models_isolated(tmp_path):
    """Same query, different model → different cache key."""
    c1 = EmbeddingCache(tmp_path / "ec", model_name="model-a")
    c2 = EmbeddingCache(tmp_path / "ec", model_name="model-b")
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)
    c1.set("query", vec_a)
    c2.set("query", vec_b)
    np.testing.assert_array_equal(c1.get("query"), vec_a)
    np.testing.assert_array_equal(c2.get("query"), vec_b)
    c1.close()
    c2.close()


def test_cache_len(tmp_path):
    cache = EmbeddingCache(tmp_path / "ec", model_name="m")
    assert len(cache) == 0
    cache.set("a", np.array([1.0], dtype=np.float32))
    cache.set("b", np.array([2.0], dtype=np.float32))
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    cache.close()


def test_cache_persists_across_instances(tmp_path):
    cache_dir = tmp_path / "ec"
    c1 = EmbeddingCache(cache_dir, model_name="m")
    c1.set("a", np.array([1.0, 2.0], dtype=np.float32))
    c1.close()

    c2 = EmbeddingCache(cache_dir, model_name="m")
    got = c2.get("a")
    np.testing.assert_array_equal(got, np.array([1.0, 2.0], dtype=np.float32))
    c2.close()


# ============================================================
# Embedder (small model — actual encode)
# ============================================================
@pytest.fixture(scope="module")
def small_embedder():
    """Tải model 1 lần per test module — small model ~17MB."""
    return make_small_embedder()


@pytest.mark.slow
def test_small_embedder_encode_returns_correct_shape(small_embedder):
    texts = ["hello world", "xin chào"]
    vectors = small_embedder.encode(texts)
    assert vectors.shape == (2, small_embedder.dim)
    assert vectors.dtype == np.float32


@pytest.mark.slow
def test_small_embedder_normalized(small_embedder):
    """Default normalize=True → mỗi vector phải unit length."""
    vec = small_embedder.encode(["xin chào"])[0]
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-3


@pytest.mark.slow
def test_small_embedder_different_text_different_vec(small_embedder):
    a = small_embedder.encode_one("hello")
    b = small_embedder.encode_one("xin chào")
    # Cosine similarity chắc chắn != 1.0 (khác nhau)
    cos = float(np.dot(a, b))
    assert cos < 0.999


@pytest.mark.slow
def test_small_embedder_empty_input_returns_empty(small_embedder):
    out = small_embedder.encode([])
    assert out.shape[0] == 0


@pytest.mark.slow
def test_embedder_dim_consistent(small_embedder):
    """`dim` property phải khớp với output shape."""
    vec = small_embedder.encode_one("test")
    assert vec.shape[0] == small_embedder.dim


# ============================================================
# Default embedder (bge-m3) — chỉ kiểm tra construction, không actually load
# ============================================================
def test_default_embedder_constructs():
    e = make_default_embedder()
    assert isinstance(e, Embedder)
    assert e.name == "BAAI/bge-m3"
    # KHÔNG gọi `.dim` hoặc `.encode` — sẽ trigger 2.27GB download.


def test_st_embedder_lazy_load():
    """Construct không trigger download."""
    e = STEmbedder(model_name="some/model-not-exist", device="cpu")
    assert e._model is None  # not loaded yet
    assert e.name == "some/model-not-exist"

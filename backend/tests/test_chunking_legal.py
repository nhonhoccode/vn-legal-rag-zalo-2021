"""Tests cho legal_chunker."""

from src.chunking.legal_chunker import (
    Chunk,
    _build_chunk_id,
    _safe_chunk_id,
    approx_token_count,
    chunk_article,
)


def _article(text: str, *, law_id: str = "45/2019/qh14", article_id: str = "41", domain: str = "labor") -> dict:
    return {
        "law_id": law_id,
        "law_title": "Bộ luật Lao động",
        "article_id": article_id,
        "text": text,
        "domain": domain,
        "metadata": {"type_full": "Luật/Bộ luật"},
    }


# ============================================================
# Token approximation
# ============================================================
def test_approx_token_count():
    assert approx_token_count("") == 1
    assert approx_token_count("a" * 4) == 1
    assert approx_token_count("a" * 400) == 100


# ============================================================
# Chunk ID slugification
# ============================================================
def test_safe_chunk_id_basic():
    assert _safe_chunk_id("45/2019/qh14", "41") == "45_2019_qh14_41"


def test_safe_chunk_id_special_chars():
    assert _safe_chunk_id("01/2009/tt-bnn", "01/2009/tt-bnn__1") == "01_2009_tt-bnn_01_2009_tt-bnn__1"


def test_build_chunk_id_with_khoan():
    assert _build_chunk_id("a_1", "1", None) == "a_1_k1"


def test_build_chunk_id_with_khoan_diem():
    assert _build_chunk_id("a_1", "2", "a") == "a_1_k2_da"


# ============================================================
# Chunking strategy: short article
# ============================================================
def test_short_article_yields_single_chunk():
    art = _article("Điều 1. Phạm vi điều chỉnh\nNội dung ngắn.")
    chunks = list(chunk_article(article=art))
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_strategy"] == "article"
    assert chunks[0].metadata["khoan_id"] is None


def test_too_short_article_dropped():
    art = _article("ngắn")  # < 20 chars
    chunks = list(chunk_article(article=art))
    assert chunks == []


def test_chunk_preserves_metadata():
    art = _article("Điều 1. Quy định nội dung article ngắn vừa đủ qua min chars.")
    chunks = list(chunk_article(article=art))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.law_id == "45/2019/qh14"
    assert c.law_title == "Bộ luật Lao động"
    assert c.domain == "labor"
    assert c.metadata.get("type_full") == "Luật/Bộ luật"
    assert "char_len" in c.metadata
    assert "token_len" in c.metadata


# ============================================================
# Chunking strategy: long article with Khoản
# ============================================================
def _long_text_with_khoan() -> str:
    """Article > 500 tokens (~2000 chars) với Khoản 1, 2, 3."""
    body = "X" * 800  # ~200 tokens per khoan
    return (
        "Điều 41. Bồi thường thiệt hại\n"
        f"1. Khi đơn phương chấm dứt hợp đồng. {body}\n"
        f"2. Người lao động được bồi thường. {body}\n"
        f"3. Trường hợp đặc biệt. {body}"
    )


def test_long_article_splits_by_khoan():
    art = _article(_long_text_with_khoan())
    chunks = list(chunk_article(article=art, max_chunk_tokens=500))
    # 3 Khoản → 3 chunks (prefix "Điều 41..." có thể là 4th nếu intro đủ dài)
    khoan_chunks = [c for c in chunks if c.metadata.get("khoan_id")]
    assert len(khoan_chunks) >= 3
    assert {c.metadata["khoan_id"] for c in khoan_chunks} >= {"1", "2", "3"}
    for c in khoan_chunks:
        assert c.metadata["chunk_strategy"] == "khoan"


def test_long_article_chunk_ids_unique():
    art = _article(_long_text_with_khoan())
    chunks = list(chunk_article(article=art, max_chunk_tokens=500))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ============================================================
# Chunking strategy: long Khoản → split by Điểm
# ============================================================
def _long_khoan_with_diem() -> str:
    # 1 Khoản với 3 Điểm a/b/c, mỗi Điểm dài
    body = "X" * 800
    return (
        "Điều 50. Quyền và nghĩa vụ.\n"
        f"1. Người lao động có các quyền sau đây:\n"
        f"a) Quyền nghỉ phép. {body}\n"
        f"b) Quyền tham gia công đoàn. {body}\n"
        f"c) Quyền yêu cầu trả lương đúng hạn. {body}"
    )


def test_long_khoan_splits_by_diem():
    art = _article(_long_khoan_with_diem())
    chunks = list(chunk_article(article=art, max_chunk_tokens=200))  # buộc split
    diem_chunks = [c for c in chunks if c.metadata.get("diem_id")]
    assert len(diem_chunks) >= 3
    assert {c.metadata["diem_id"] for c in diem_chunks} >= {"a", "b", "c"}


# ============================================================
# Edge case: long article without Khoản markers
# ============================================================
def test_long_article_no_khoan_kept_as_single():
    """Không có "1. " markers → emit nguyên chunk + flag."""
    text = "Điều X. " + ("Đây là nội dung không có Khoản markers. " * 100)
    art = _article(text)
    chunks = list(chunk_article(article=art, max_chunk_tokens=100))
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_strategy"] == "article_oversize_no_khoan"


# ============================================================
# Output is Chunk dataclass with to_dict
# ============================================================
def test_chunk_to_dict_contains_required_fields():
    art = _article("Điều 1. Quy định ngắn vừa đủ chars min.")
    chunks = list(chunk_article(article=art))
    d = chunks[0].to_dict()
    assert {"chunk_id", "law_id", "law_title", "article_id", "text", "domain", "metadata"} <= d.keys()


def test_chunk_dataclass_slots():
    c = Chunk(
        chunk_id="x",
        law_id="L",
        law_title="t",
        article_id="1",
        text="x",
        domain="labor",
    )
    # slots=True
    assert not hasattr(c, "__dict__")


# ============================================================
# False positive guard: "1.5%" không match Khoản
# ============================================================
def test_decimal_numbers_not_treated_as_khoan():
    text = (
        "Điều X. Quy định lãi suất 1.5% và 2.5% cho khoản vay. "
        + ("Đây là nội dung kéo dài chữ này nội dung kéo dài chữ này. " * 50)
    )
    art = _article(text)
    chunks = list(chunk_article(article=art, max_chunk_tokens=500))
    # Không có Khoản marker hợp lệ → 1 chunk (article hoặc oversize_no_khoan)
    khoan_chunks = [c for c in chunks if c.metadata.get("khoan_id")]
    assert len(khoan_chunks) == 0

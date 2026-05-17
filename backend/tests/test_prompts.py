"""Tests cho prompts module."""

from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_messages,
    build_user_prompt,
    is_refusal,
    parse_citations,
)
from src.vectorstore.base import SearchHit


def _hit(law_id="45/2019/qh14", law_title="Bộ luật Lao động", article_id="41",
         text="Bồi thường thiệt hại...", khoan_id=None, score=0.8) -> SearchHit:
    meta = {"law_id": law_id, "law_title": law_title, "article_id": article_id}
    if khoan_id:
        meta["khoan_id"] = khoan_id
    return SearchHit(chunk_id=f"{law_id}_{article_id}", text=text, score=score, metadata=meta)


# ============================================================
# build_context_block
# ============================================================
def test_context_block_numbers_chunks():
    hits = [_hit(article_id="1"), _hit(article_id="2"), _hit(article_id="3")]
    block = build_context_block(hits)
    assert "[Văn bản 1:" in block
    assert "[Văn bản 2:" in block
    assert "[Văn bản 3:" in block


def test_context_block_includes_khoan_when_present():
    hits = [_hit(article_id="13", khoan_id="2")]
    block = build_context_block(hits)
    assert "Điều 13" in block
    assert "Khoản 2" in block


def test_context_block_excludes_khoan_when_none():
    hits = [_hit(article_id="13", khoan_id=None)]
    block = build_context_block(hits)
    assert "Điều 13" in block
    assert "Khoản" not in block


def test_context_block_separator():
    hits = [_hit(article_id="1"), _hit(article_id="2")]
    block = build_context_block(hits)
    assert "---" in block


def test_context_block_truncates_at_budget():
    long_text = "x" * 10000
    hits = [_hit(text=long_text, article_id=str(i)) for i in range(20)]
    block = build_context_block(hits, max_chars=5000)
    assert len(block) <= 6000  # some slop for headers


def test_context_block_empty_hits():
    assert build_context_block([]) == ""


# ============================================================
# build_user_prompt + build_messages
# ============================================================
def test_user_prompt_has_query_and_context():
    hits = [_hit()]
    prompt = build_user_prompt("Câu hỏi?", hits)
    assert "Câu hỏi?" in prompt
    assert "VĂN BẢN PHÁP LUẬT" in prompt
    assert "TRẢ LỜI" in prompt


def test_user_prompt_no_hits_says_no_context():
    prompt = build_user_prompt("Câu hỏi?", [])
    assert "Không có văn bản" in prompt


def test_build_messages_has_system_and_user():
    msgs = build_messages("test", [_hit()])
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[0].content == SYSTEM_PROMPT
    assert msgs[1].role == "user"


# ============================================================
# parse_citations
# ============================================================
def test_parse_simple_citation():
    hits = [_hit(article_id="41")]
    answer = "Theo [Văn bản 1, Điều 41], người lao động được bồi thường."
    cits = parse_citations(answer, hits)
    assert len(cits) == 1
    assert cits[0].text_index == 1
    assert cits[0].article_id == "41"
    assert cits[0].khoan_id is None
    assert cits[0].matched is True
    assert cits[0].law_id == "45/2019/qh14"


def test_parse_citation_with_khoan():
    hits = [_hit(article_id="41")]
    answer = "Theo [Văn bản 1, Điều 41, Khoản 2], ..."
    cits = parse_citations(answer, hits)
    assert len(cits) == 1
    assert cits[0].khoan_id == "2"


def test_parse_multiple_citations():
    hits = [_hit(article_id="41"), _hit(article_id="13")]
    answer = "[Văn bản 1, Điều 41] và [Văn bản 2, Điều 13]"
    cits = parse_citations(answer, hits)
    assert len(cits) == 2
    assert cits[0].text_index == 1
    assert cits[1].text_index == 2


def test_parse_out_of_range_index_unmatched():
    hits = [_hit()]
    answer = "[Văn bản 5, Điều 99]"
    cits = parse_citations(answer, hits)
    assert len(cits) == 1
    assert cits[0].matched is False
    assert cits[0].law_id == ""


def test_parse_no_citations_in_answer():
    hits = [_hit()]
    cits = parse_citations("Câu trả lời không có citation.", hits)
    assert cits == []


def test_parse_case_insensitive():
    hits = [_hit(article_id="41")]
    answer = "[văn bản 1, điều 41]"  # lowercase
    cits = parse_citations(answer, hits)
    assert len(cits) == 1


# ============================================================
# is_refusal
# ============================================================
def test_refusal_detected():
    assert is_refusal("Tôi không tìm thấy thông tin này trong các văn bản đã được cung cấp.")
    assert is_refusal("Trả lời: Không tìm thấy thông tin.")


def test_refusal_not_in_normal_answer():
    assert not is_refusal("Theo Điều 41, người lao động được bồi thường.")

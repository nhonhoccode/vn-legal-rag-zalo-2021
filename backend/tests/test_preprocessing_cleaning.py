"""Tests cho cleaning — repeating dots/dashes, page markers."""

from src.preprocessing.cleaning import (
    clean,
    is_too_short,
    remove_dots_only_lines,
    remove_page_markers,
    remove_repeating_punct,
)


def test_remove_repeating_dots():
    assert remove_repeating_punct("a....b") == "a...b"
    assert remove_repeating_punct("a.....b") == "a...b"
    # 3 dots preserved (ellipsis có nghĩa)
    assert remove_repeating_punct("a...b") == "a...b"


def test_remove_repeating_dashes():
    assert remove_repeating_punct("a----b") == "a—b"
    # 4 em-dashes → collapse
    assert remove_repeating_punct("a————b") == "a—b"
    # 3 em-dashes giữ nguyên (chỉ 4+ mới collapse)
    assert remove_repeating_punct("a———b") == "a———b"
    # 3 dashes giữ nguyên
    assert remove_repeating_punct("a---b") == "a---b"


def test_remove_page_marker():
    text = "Nội dung\nTrang 5\nTiếp theo"
    cleaned = remove_page_markers(text)
    assert "Trang 5" not in cleaned
    assert "Nội dung" in cleaned
    assert "Tiếp theo" in cleaned


def test_remove_page_marker_with_total():
    text = "Nội dung\nTrang 5/10\nTiếp"
    cleaned = remove_page_markers(text)
    assert "Trang 5/10" not in cleaned


def test_remove_page_marker_does_not_remove_inline_trang():
    text = "Trang web https://abc.com là nguồn"
    cleaned = remove_page_markers(text)
    assert cleaned == text  # not standalone marker


def test_remove_dots_only_lines():
    text = "Nội dung\n....\nTiếp"
    assert remove_dots_only_lines(text).count("....") == 0


def test_clean_pipeline():
    text = "Điều 1...\n....\nTrang 1\nNội dung."
    result = clean(text)
    assert "Điều 1..." in result
    assert "Trang 1" not in result
    # 3-dot ellipsis sau Điều 1 giữ nguyên (chỉ 4+ dots mới collapse)


def test_is_too_short_threshold():
    assert is_too_short("abc") is True
    assert is_too_short("a" * 19) is True
    assert is_too_short("a" * 20) is False
    assert is_too_short("Đây là một đoạn text dài hơn 20 chars") is False


def test_is_too_short_strips():
    assert is_too_short("   abc   ") is True  # strip xong < 20
    assert is_too_short("   " + "a" * 25 + "   ") is False

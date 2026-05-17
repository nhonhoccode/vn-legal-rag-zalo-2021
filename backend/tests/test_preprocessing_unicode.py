"""Tests cho unicode_norm — NFC + zero-width + whitespace."""

import unicodedata

from src.preprocessing.unicode_norm import (
    full_normalize,
    nfc_normalize,
    normalize_whitespace,
    remove_zero_width,
)


def test_nfc_normalize_combines_decomposed():
    """`o` + combining grave (\\u0300) → `ò` (precomposed)."""
    decomposed = "ò"  # NFD form
    assert decomposed != "ò"  # different codepoints
    normalized = nfc_normalize(decomposed)
    assert normalized == "ò"
    assert unicodedata.normalize("NFC", normalized) == normalized  # idempotent


def test_nfc_normalize_vietnamese_words():
    # "ngôn ngữ" with mixed forms
    decomposed = "ngôn ngữ"
    normalized = nfc_normalize(decomposed)
    assert normalized == "ngôn ngữ"


def test_nfc_idempotent_on_already_composed():
    text = "Bộ luật Lao động"
    assert nfc_normalize(text) == text


def test_remove_zero_width_chars():
    text = "Bộ​luật‌Lao‍động﻿"
    assert remove_zero_width(text) == "BộluậtLaođộng"


def test_remove_zero_width_preserves_normal_chars():
    text = "Bộ luật Lao động"  # no zero-width
    assert remove_zero_width(text) == text


def test_normalize_whitespace_crlf_to_lf():
    assert normalize_whitespace("a\r\nb") == "a\nb"
    assert normalize_whitespace("a\rb") == "a\nb"


def test_normalize_whitespace_collapses_multi_newlines():
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"
    assert normalize_whitespace("a\n\nb") == "a\n\nb"  # 2 newlines preserved


def test_normalize_whitespace_strips_trailing_each_line():
    assert normalize_whitespace("a   \nb") == "a\nb"


def test_normalize_whitespace_collapses_multi_spaces():
    assert normalize_whitespace("a    b") == "a b"
    assert normalize_whitespace("a\t\tb") == "a b"


def test_normalize_whitespace_strips_outer():
    assert normalize_whitespace("  hello  ") == "hello"


def test_full_normalize_pipeline():
    text = "  Bộ​luật\r\n\r\n\r\nLao   động  "
    result = full_normalize(text)
    assert result == "Bộluật\n\nLao động"


def test_full_normalize_preserves_paragraph_break():
    text = "Điều 1.\n\nNội dung điều 1."
    assert full_normalize(text) == "Điều 1.\n\nNội dung điều 1."

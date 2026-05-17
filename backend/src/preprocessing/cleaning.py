"""Cleaning rules cho text văn bản pháp luật VN."""

from __future__ import annotations

import re

# Pattern noise phổ biến trong văn bản luật.

# "...", "....", "....." — repeating dots (thường là filler, không có nghĩa).
_REPEATING_DOTS_RE = re.compile(r"\.{4,}")

# "—" Em-dash repeating (sometimes from PDF layout)
_REPEATING_DASH_RE = re.compile(r"[-—]{4,}")

# Bare page numbers on their own line: "Trang 1", "Trang 1/10", or just "12"
# Conservative: only match "Trang N" — bare numbers ambiguous (có thể là Khoản N).
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*Trang\s+\d+(\s*/\s*\d+)?\s*$", re.MULTILINE)

# Dòng chỉ chứa "..." thường là filler.
_DOTS_ONLY_LINE_RE = re.compile(r"^\s*[.…]+\s*$", re.MULTILINE)


def remove_repeating_punct(text: str) -> str:
    """Replace dots/dashes 4+ liên tiếp bằng 3 dots / 1 dash."""
    text = _REPEATING_DOTS_RE.sub("...", text)
    text = _REPEATING_DASH_RE.sub("—", text)
    return text


def remove_page_markers(text: str) -> str:
    """Xoá dòng marker trang ('Trang 1', 'Trang 1/10')."""
    return _PAGE_NUMBER_LINE_RE.sub("", text)


def remove_dots_only_lines(text: str) -> str:
    """Xoá dòng chỉ có dots."""
    return _DOTS_ONLY_LINE_RE.sub("", text)


def clean(text: str) -> str:
    """Pipeline cleaning đầy đủ. Áp dụng SAU `unicode_norm.full_normalize`."""
    text = remove_repeating_punct(text)
    text = remove_page_markers(text)
    text = remove_dots_only_lines(text)
    return text


def is_too_short(text: str, min_chars: int = 20) -> bool:
    """Article quá ngắn (< 20 chars) thường là noise (header rỗng, ...)."""
    return len(text.strip()) < min_chars

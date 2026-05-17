"""Unicode normalization cho tiếng Việt.

Bắt buộc cho mọi text trước khi index/embed/search vì tiếng Việt có
**2 cách encode dấu** (precomposed vs combining), không normalize → match sai.

Reference: DATA_STRATEGY.md §10.3.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width characters thường bị crawl/copy-paste lẫn vào — phải remove.
_ZERO_WIDTH_CHARS = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "⁠",  # WORD JOINER
    "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
)
_ZERO_WIDTH_RE = re.compile("|".join(_ZERO_WIDTH_CHARS))

# Tab và carriage return → space / newline (chuẩn hoá whitespace).
_CR_RE = re.compile(r"\r\n?")
# Multiple consecutive newlines → max 2 (giữ block paragraph).
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Trailing whitespace mỗi dòng.
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
# Multiple consecutive spaces (in một line) → 1 space.
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def nfc_normalize(text: str) -> str:
    """Apply Unicode NFC normalization.

    Tiếng Việt: chữ "ò" có thể là `\\u00f2` (precomposed) hoặc `\\u006f\\u0300`
    (o + combining grave). NFC merge thành dạng precomposed.
    """
    return unicodedata.normalize("NFC", text)


def remove_zero_width(text: str) -> str:
    """Xoá các ký tự zero-width (BOM, ZWSP, ...)."""
    return _ZERO_WIDTH_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hoá whitespace:

    - CR/CRLF → LF
    - 3+ newlines → 2
    - Trailing whitespace mỗi dòng → bỏ
    - 2+ spaces/tabs liền nhau → 1 space
    - Strip leading/trailing
    """
    text = _CR_RE.sub("\n", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def full_normalize(text: str) -> str:
    """Pipeline đầy đủ: NFC → remove zero-width → normalize whitespace."""
    text = nfc_normalize(text)
    text = remove_zero_width(text)
    text = normalize_whitespace(text)
    return text

"""Chunking văn bản pháp luật VN — chunk-by-Điều với fallback theo Khoản.

Theo `RETRIEVAL_STRATEGY.md` §1 và `DECISIONS.md` D5:

- Mỗi article (Điều) là 1 chunk tự nhiên.
- Nếu Điều > ``max_chunk_tokens`` (default 500) → split theo Khoản (`^\\d+\\.`).
- Nếu Khoản vẫn dài → split theo Điểm (`^[a-zđ]\\)`).
- Article < ``min_chunk_chars`` (default 20): drop hoặc giữ với flag `is_short=True`.

Token approximation: 1 token ≈ 4 chars (heuristic Vietnamese; chính xác hơn cần
sentence-piece tokenizer của embedding model — implement Phase 5 nếu cần).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

# Khoản pattern: line bắt đầu bằng "1. " "2. " ... "10. "
# Chú ý: KHÔNG match "1.5" hoặc "10.000" (số thập phân / số tiền).
_KHOAN_RE = re.compile(r"(?m)^(\d{1,2})\.\s+")

# Điểm pattern: line bắt đầu bằng "a) " "b) " ... với chữ thường VN (a-z + đ).
_DIEM_RE = re.compile(r"(?m)^([a-zđ])\)\s+")

# Điều header — dùng để strip nếu có trong text (article body lẽ ra không kèm header).
_DIEU_HEADER_RE = re.compile(r"^Điều\s+\d+[a-z]?\.\s*[^\n]*\n?")


def approx_token_count(text: str) -> int:
    """Heuristic: 1 token ≈ 4 chars cho tiếng Việt."""
    return max(1, len(text) // 4)


@dataclass(slots=True)
class Chunk:
    """1 chunk sẵn sàng để embed."""

    chunk_id: str
    law_id: str
    law_title: str
    article_id: str
    text: str
    domain: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_by_pattern(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Split text bằng regex pattern. Returns list of (marker, segment_text).

    Ví dụ với KHOAN_RE và text ``"abc\n1. xxx\n2. yyy"``:
        [("", "abc\n"), ("1", "xxx\n"), ("2", "yyy")]
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    segments: list[tuple[str, str]] = []
    # Prefix trước marker đầu tiên
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            segments.append(("", prefix))

    for i, m in enumerate(matches):
        marker = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment_text = text[start:end].strip()
        if segment_text:
            segments.append((marker, segment_text))

    return segments


def chunk_article(
    *,
    article: dict,
    max_chunk_tokens: int = 500,
    min_chunk_chars: int = 20,
) -> Iterator[Chunk]:
    """Chunk 1 article (output từ Phase 3) → 1 hoặc nhiều Chunk.

    Args:
        article: dict từ articles_clean.jsonl (`law_id`, `law_title`, `article_id`,
                 `text`, `domain`, `metadata`).
        max_chunk_tokens: token threshold để trigger split theo Khoản.
        min_chunk_chars: min length sau strip để keep chunk.

    Yields:
        Chunk objects.
    """
    text = (article.get("text") or "").strip()
    if len(text) < min_chunk_chars:
        return

    law_id = article.get("law_id", "")
    law_title = article.get("law_title", "")
    article_id = article.get("article_id", "")
    domain = article.get("domain", "unknown")
    base_metadata = article.get("metadata", {}) or {}

    base_chunk_id = _safe_chunk_id(law_id, article_id)

    # Trường hợp 1: article đủ ngắn → 1 chunk.
    if approx_token_count(text) <= max_chunk_tokens:
        yield Chunk(
            chunk_id=base_chunk_id,
            law_id=law_id,
            law_title=law_title,
            article_id=article_id,
            text=text,
            domain=domain,
            metadata={
                **base_metadata,
                "chunk_strategy": "article",
                "khoan_id": None,
                "diem_id": None,
                "char_len": len(text),
                "token_len": approx_token_count(text),
            },
        )
        return

    # Trường hợp 2: dài → split theo Khoản.
    khoan_segments = _split_by_pattern(text, _KHOAN_RE)
    if len(khoan_segments) <= 1:
        # No Khoản markers detected → emit as single chunk (giữ nguyên), warn-able sau.
        yield Chunk(
            chunk_id=base_chunk_id,
            law_id=law_id,
            law_title=law_title,
            article_id=article_id,
            text=text,
            domain=domain,
            metadata={
                **base_metadata,
                "chunk_strategy": "article_oversize_no_khoan",
                "khoan_id": None,
                "diem_id": None,
                "char_len": len(text),
                "token_len": approx_token_count(text),
            },
        )
        return

    # Có Khoản → emit per khoan.
    # Prefix (text trước Khoản 1, thường là body intro của Điều) cũng emit.
    for khoan_id, khoan_text in khoan_segments:
        if len(khoan_text) < min_chunk_chars:
            continue

        # Re-compose with khoan header so context preserved.
        full_text = f"{khoan_id}. {khoan_text}" if khoan_id else khoan_text

        # Sub-case: Khoản vẫn quá dài → split theo Điểm.
        if approx_token_count(full_text) > max_chunk_tokens:
            diem_segments = _split_by_pattern(khoan_text, _DIEM_RE)
            if len(diem_segments) > 1:
                for diem_id, diem_text in diem_segments:
                    if len(diem_text) < min_chunk_chars:
                        continue
                    final_text = (
                        f"{khoan_id}. ... {diem_id}) {diem_text}" if diem_id else diem_text
                    )
                    chunk_id = _build_chunk_id(base_chunk_id, khoan_id, diem_id)
                    yield Chunk(
                        chunk_id=chunk_id,
                        law_id=law_id,
                        law_title=law_title,
                        article_id=article_id,
                        text=final_text,
                        domain=domain,
                        metadata={
                            **base_metadata,
                            "chunk_strategy": "khoan_diem",
                            "khoan_id": khoan_id or None,
                            "diem_id": diem_id or None,
                            "char_len": len(final_text),
                            "token_len": approx_token_count(final_text),
                        },
                    )
                continue
            # Không có Điểm → emit nguyên Khoản (oversize, accepted).

        chunk_id = _build_chunk_id(base_chunk_id, khoan_id, None)
        yield Chunk(
            chunk_id=chunk_id,
            law_id=law_id,
            law_title=law_title,
            article_id=article_id,
            text=full_text,
            domain=domain,
            metadata={
                **base_metadata,
                "chunk_strategy": "khoan",
                "khoan_id": khoan_id or None,
                "diem_id": None,
                "char_len": len(full_text),
                "token_len": approx_token_count(full_text),
            },
        )


def _safe_chunk_id(law_id: str, article_id: str) -> str:
    """Build chunk id sạch (slugify) từ law_id + article_id."""
    base = f"{law_id}_{article_id}".lower()
    # Replace non-alphanumeric (giữ _ và -) bằng _.
    return re.sub(r"[^a-z0-9_\-]", "_", base)


def _build_chunk_id(base: str, khoan_id: str | None, diem_id: str | None) -> str:
    parts = [base]
    if khoan_id:
        parts.append(f"k{khoan_id}")
    if diem_id:
        parts.append(f"d{diem_id}")
    return "_".join(parts)

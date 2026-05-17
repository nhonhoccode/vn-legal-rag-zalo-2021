"""Vietnamese legal RAG prompt templates với strict citation rules.

Theo `RETRIEVAL_STRATEGY.md` §10–11:
- System prompt: refuse hallucination, force citation, low temperature.
- Context format: numbered chunks với metadata (số văn bản, Điều, Khoản).

Phase 9 thêm:
- Standalone query rewriting prompt (resolve references trong multi-turn).
- Multi-turn user prompt với history context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.vectorstore.base import SearchHit

from .conversation import Turn
from .llm_client import LLMMessage

# Token budget cho context (~6k tokens free cho query + answer).
# Heuristic 4 chars/token cho VN → ~24k chars context max.
MAX_CONTEXT_CHARS = 24000


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về pháp luật Việt Nam.

QUY TẮC TRẢ LỜI:
1. Ưu tiên trả lời dựa trên các văn bản pháp luật được cung cấp ở phần "VĂN BẢN PHÁP LUẬT" bên dưới.

2. PHÂN BIỆT các trường hợp:
   a) Văn bản TRẢ LỜI TRỰC TIẾP câu hỏi → trả lời đầy đủ, trích dẫn rõ ràng.
   b) Văn bản LIÊN QUAN nhưng không trả lời trực tiếp → trả lời dựa trên thông tin có, KÈM lưu ý:
      "Lưu ý: Các văn bản dưới đây liên quan đến chủ đề bạn hỏi, nhưng có thể chưa phải điều luật gốc trực tiếp. Bạn nên tham khảo thêm để có thông tin chính xác."
   c) MULTI-TURN — user yêu cầu giải thích/mở rộng/làm rõ (vd: "giải thích cụ thể hơn", "còn trường hợp khác", "ví dụ"):
      → ĐƯỢC PHÉP dựa vào câu trả lời TRƯỚC trong lịch sử hội thoại để diễn giải sâu hơn,
      → KHÔNG cần văn bản mới phải khớp 100% — chỉ cần liên quan đến chủ đề đã thảo luận.
      → Giữ citation từ câu trước nếu phù hợp.
   d) KHÔNG văn bản nào liên quan VÀ không có context lịch sử → trả lời:
      "Tôi không tìm thấy thông tin này trong các văn bản đã được cung cấp. Bạn thử diễn đạt câu hỏi khác hoặc cụ thể hơn nhé."

3. TUYỆT ĐỐI KHÔNG bịa số văn bản, số Điều, số Khoản mới ngoài context hoặc lịch sử.

4. Trích dẫn format: [Văn bản N, Điều X] hoặc [Văn bản N, Điều X, Khoản Y].
   Trong đó N là số thứ tự văn bản trong danh sách bên dưới.

5. Trả lời ngắn gọn, đúng trọng tâm, tiếng Việt rõ ràng. Tối đa 4-5 câu trừ khi câu hỏi yêu cầu chi tiết.
"""


def build_context_block(hits: list[SearchHit], *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format các chunks thành block context có numbered references.

    Mỗi chunk có header `[Văn bản N: <law_title>, Điều <article_id>]`.
    Truncate nếu vượt budget.
    """
    parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, 1):
        law_id = hit.metadata.get("law_id", "?")
        law_title = hit.metadata.get("law_title", "") or law_id
        article_id = hit.metadata.get("article_id", "?")
        khoan_id = hit.metadata.get("khoan_id")

        header_parts = [f"[Văn bản {i}: {law_title} (số {law_id})", f"Điều {article_id}"]
        if khoan_id:
            header_parts.append(f"Khoản {khoan_id}")
        header = ", ".join(header_parts) + "]"

        block = f"{header}\n{hit.text}"
        if total_chars + len(block) > max_chars:
            # Truncate this block to fit budget.
            remaining = max_chars - total_chars - len(header) - 10
            if remaining > 50:
                block = f"{header}\n{hit.text[:remaining]}...(truncated)"
                parts.append(block)
            break
        parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(parts)


def build_user_prompt(query: str, hits: list[SearchHit]) -> str:
    """Build the user message: context + question."""
    if hits:
        context = build_context_block(hits)
        return f"""=== VĂN BẢN PHÁP LUẬT ===

{context}

=== CÂU HỎI ===

{query}

=== TRẢ LỜI ===
"""
    return f"""=== VĂN BẢN PHÁP LUẬT ===

(Không có văn bản nào được truy xuất.)

=== CÂU HỎI ===

{query}

=== TRẢ LỜI ===
"""


def build_messages(query: str, hits: list[SearchHit]) -> list[LLMMessage]:
    """Build full message list cho LLM call (single-turn)."""
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=build_user_prompt(query, hits)),
    ]


# ============================================================
# Multi-turn (Phase 9)
# ============================================================

# Tóm tắt history trước đó (nếu có) để inject vào user prompt.
# KHÔNG dùng role="assistant" history vì system prompt strict citation rules
# có thể conflict với answer cũ → an toàn hơn là tóm tắt trong user message.
def build_messages_multiturn(
    query: str,
    hits: list[SearchHit],
    history: list[Turn] | None = None,
) -> list[LLMMessage]:
    """Build messages với conversation history.

    Cấu trúc:
        [system, history[0], history[1], ..., user(query+context)]

    History được pass như role assistant/user thực sự để LLM track context.
    """
    messages: list[LLMMessage] = [LLMMessage(role="system", content=SYSTEM_PROMPT)]

    if history:
        for turn in history:
            messages.append(LLMMessage(role=turn.role, content=turn.content))

    messages.append(LLMMessage(role="user", content=build_user_prompt(query, hits)))
    return messages


# ============================================================
# Standalone query rewriting (Phase 9)
# ============================================================

REWRITE_SYSTEM_PROMPT = """Bạn là trợ lý đọc lịch sử hội thoại để viết lại câu hỏi mới thành câu hỏi độc lập (standalone) — phục vụ retrieval pháp luật Việt Nam.

QUY TẮC:
1. Đọc lịch sử + câu hỏi mới của user.
2. Nếu câu hỏi mới phụ thuộc vào context của lịch sử (vd có "câu trên", "điều đó", "vậy", "còn", "tương tự", "giải thích cụ thể hơn", "ví dụ", "làm rõ"...):
   → Viết lại câu hỏi GIỮ NGUYÊN CHỦ ĐỀ pháp lý từ lịch sử + diễn đạt ý mới của user.
   → Ví dụ: nếu lịch sử bàn về "chế độ thai sản của lao động nữ mang thai hộ" và user hỏi "giải thích cụ thể hơn"
     → rewrite thành "Chế độ thai sản của lao động nữ mang thai hộ được quy định cụ thể như thế nào, gồm những quyền lợi gì?"
3. Nếu câu hỏi mới đã đủ độc lập → trả về NGUYÊN VĂN.
4. CHỈ trả về câu hỏi đã viết lại, không thêm giải thích, không thêm dấu ngoặc kép.
5. Giữ ngôn ngữ tiếng Việt, dùng thuật ngữ pháp lý chuẩn.
"""


def build_rewrite_messages(query: str, history: list[Turn]) -> list[LLMMessage]:
    """Build messages cho standalone query rewriting call.

    History format gọn: chỉ liệt kê các turn dạng "User: ... / Assistant: ...".
    """
    history_text = _format_history_for_rewrite(history)
    user_msg = f"""=== LỊCH SỬ HỘI THOẠI ===

{history_text}

=== CÂU HỎI MỚI ===

{query}

=== CÂU HỎI ĐỘC LẬP ===
"""
    return [
        LLMMessage(role="system", content=REWRITE_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]


def _format_history_for_rewrite(history: list[Turn], max_chars: int = 2000) -> str:
    """Compact format cho rewrite step: chỉ user/assistant text, không context blocks."""
    if not history:
        return "(Không có lịch sử.)"

    lines: list[str] = []
    total = 0
    # Render từ NEW → OLD để giữ những turn mới nhất nếu cần truncate.
    rendered: list[str] = []
    for turn in reversed(history):
        prefix = "User: " if turn.role == "user" else "Assistant: "
        line = f"{prefix}{turn.content}"
        if total + len(line) > max_chars:
            break
        rendered.append(line)
        total += len(line)

    # Reverse lại để order = chronological (old → new).
    lines = list(reversed(rendered))
    return "\n".join(lines)


# Heuristic detect query có phụ thuộc reference không (tránh gọi LLM rewrite không cần thiết).
_REFERENCE_PATTERNS = re.compile(
    r"\b(còn|vậy|thế|đó|này|ấy|trên|trước|tương\s+tự|như\s+vậy|tiếp\s+theo|cụ\s+thể)\b",
    re.IGNORECASE,
)


def query_likely_needs_rewrite(query: str) -> bool:
    """Heuristic: query có chứa reference words → likely cần rewrite.

    KHÔNG bullet-proof — chỉ optimization. Multi-turn pipeline có thể skip rewrite call
    khi query không match.
    """
    if len(query) < 5:
        return False
    return bool(_REFERENCE_PATTERNS.search(query))


# ============================================================
# Citation extraction
# ============================================================

# Match `[Văn bản N, Điều X]` hoặc `[Văn bản N, Điều X, Khoản Y]`.
_CITATION_RE = re.compile(
    r"\[Văn\s+bản\s+(\d+)(?:,\s*Điều\s+([\d\w\.]+))?(?:,\s*Khoản\s+([\d\w]+))?\]",
    re.IGNORECASE,
)

# Refuse pattern — answer chứa câu refuse → confidence thấp.
_REFUSE_PATTERN = re.compile(
    r"không\s+tìm\s+thấy\s+thông\s+tin", re.IGNORECASE
)


@dataclass(slots=True)
class ParsedCitation:
    """Citation đã parse từ LLM answer."""

    text_index: int  # số thứ tự văn bản trong context (1-based)
    article_id: str  # text như LLM viết
    khoan_id: str | None
    # Resolved fields sau khi match với hits gốc:
    law_id: str = ""
    law_title: str = ""
    matched: bool = False  # True nếu citation match một hit trong context


def parse_citations(answer: str, hits: list[SearchHit]) -> list[ParsedCitation]:
    """Extract citations từ answer + match với hits.

    `text_index` 1-based, tương ứng vị trí trong `hits`. Citation tới index
    out-of-range vẫn return nhưng `matched=False`.
    """
    citations: list[ParsedCitation] = []
    for m in _CITATION_RE.finditer(answer):
        idx_str, art_id, khoan_id = m.group(1), m.group(2), m.group(3)
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        c = ParsedCitation(
            text_index=idx,
            article_id=art_id or "",
            khoan_id=khoan_id or None,
        )

        if 1 <= idx <= len(hits):
            hit = hits[idx - 1]
            c.law_id = hit.metadata.get("law_id", "")
            c.law_title = hit.metadata.get("law_title", "") or c.law_id
            # Light validation: article_id LLM viết có khớp metadata không?
            hit_article = str(hit.metadata.get("article_id", ""))
            if not art_id or art_id in hit_article or hit_article in art_id:
                c.matched = True

        citations.append(c)

    return citations


def is_refusal(answer: str) -> bool:
    """Detect xem LLM có refuse trả lời không (theo system prompt rule 2)."""
    return bool(_REFUSE_PATTERN.search(answer))

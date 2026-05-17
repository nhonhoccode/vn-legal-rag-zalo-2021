"""Intent classification: tách chit-chat / meta-questions khỏi legal queries.

Mục tiêu: tránh chạy full RAG pipeline (retrieval + reranker) cho những câu hỏi
không cần văn bản pháp luật (chào hỏi, hỏi về assistant, cảm ơn, ...).

Chit-chat sẽ vẫn được trả lời bởi LLM với system prompt riêng — không hardcode
response để giữ tone tự nhiên + có thể thay đổi.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    LEGAL_QUERY = "legal_query"
    CHITCHAT = "chitchat"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


# Pattern matchers — keep tight to avoid false positives on legal queries.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello+|hey+|chào|xin\s*chào|chào\s*(bạn|anh|chị|em|mọi\s*người)|"
    r"good\s*(morning|afternoon|evening)|alo+)[\s!?.,]*$",
    re.IGNORECASE,
)

# Meta-questions về assistant — phải chứa "bạn"/"you"/"tôi có thể" + ngắn (<60 chars)
_META_RE = re.compile(
    r"\b("
    r"bạn\s+(là|tên|làm|có\s+thể|giúp)|"
    r"who\s+are\s+you|what\s+(can|do)\s+you|"
    r"giúp\s+(gì|được\s+gì)|làm\s+(gì|được\s+gì)|"
    r"hỗ\s+trợ\s+(gì|được\s+gì)|"
    r"chức\s+năng"
    r")\b",
    re.IGNORECASE,
)

# Acknowledgements
_ACK_RE = re.compile(
    r"^\s*(cảm\s*ơn|cám\s*ơn|thanks?|thank\s+you|thx|ok+|okay|được|tốt|hay)[\s!?.,]*$",
    re.IGNORECASE,
)

MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 1000


def classify_intent(query: str) -> Intent:
    """Classify query intent. Cheap regex-based — no LLM call."""
    q = (query or "").strip()
    n = len(q)
    # Chit-chat patterns check FIRST (hi/ok are 2 chars, valid chit-chat)
    if _GREETING_RE.match(q):
        return Intent.CHITCHAT
    if _ACK_RE.match(q):
        return Intent.CHITCHAT
    if n < MIN_QUERY_CHARS:
        return Intent.TOO_SHORT
    if n > MAX_QUERY_CHARS:
        return Intent.TOO_LONG
    # Meta-questions: phải ngắn + match pattern. Tránh false-positive với
    # câu hỏi luật dài như "Tôi có thể yêu cầu bồi thường khi...".
    if n < 60 and _META_RE.search(q):
        return Intent.CHITCHAT

    return Intent.LEGAL_QUERY


CHITCHAT_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý cho luật Việt Nam, chuyên 3 ngành: Lao động, Hình sự, Doanh nghiệp/Đầu tư.

NGỮ CẢNH HIỆN TẠI: User đang nói chuyện xã giao hoặc hỏi về khả năng của bạn (KHÔNG phải câu hỏi pháp lý cụ thể).

QUY TẮC TRẢ LỜI:
1. Trả lời tự nhiên, thân thiện, ngắn gọn (2-4 câu).
2. KHÔNG bịa thông tin pháp lý, KHÔNG dẫn điều luật.
3. Nếu user chào hỏi → chào lại + giới thiệu ngắn về khả năng.
4. Nếu user cảm ơn → đáp lễ + khuyến khích hỏi tiếp.
5. Nếu user hỏi về bạn → giới thiệu: tra cứu pháp luật VN, 3 ngành, multi-turn.
6. Có thể gợi ý 1-2 ví dụ câu hỏi pháp lý nếu phù hợp.
7. Tiếng Việt tự nhiên, có thể dùng emoji nhẹ nhàng.
"""


TOO_SHORT_MESSAGE = "Câu hỏi quá ngắn. Bạn vui lòng diễn đạt cụ thể hơn để tôi tra cứu chính xác nhé."
TOO_LONG_MESSAGE = (
    "Câu hỏi quá dài (>1000 ký tự). Bạn rút gọn lại để tôi tập trung vào ý chính."
)

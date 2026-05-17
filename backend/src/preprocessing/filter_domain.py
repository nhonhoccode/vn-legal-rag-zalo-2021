"""Filter articles theo 3 ngành luật (Decision 20).

3 ngành focus: **Lao động**, **Doanh nghiệp/Đầu tư**, **Hình sự**.

Strategy 2-tier:

1. **Whitelist law_id** (high precision): các bộ luật chính + sửa đổi đã biết.
2. **Keyword scan** trong text (recall fallback): scan first N chars cho keywords domain-specific.

Article match cả 2 → label đúng. Match nhiều domains → pick max keyword count.
Không match gì → ``None`` → drop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Domain = Literal["labor", "business", "criminal"]

# ============================================================
# Tier 1: Whitelist law_ids (chính + Bộ luật + sửa đổi)
# ============================================================

# Lao động: Bộ luật Lao động + sửa đổi.
LABOR_LAW_IDS: frozenset[str] = frozenset({
    "10/2012/qh13",   # BLLĐ 2012
    "45/2019/qh14",   # BLLĐ 2019 (current)
})

# Hình sự: Bộ luật Hình sự + sửa đổi.
CRIMINAL_LAW_IDS: frozenset[str] = frozenset({
    "100/2015/qh13",  # BLHS 2015
    "12/2017/qh14",   # Sửa đổi BLHS 2017
})

# Doanh nghiệp + Đầu tư.
BUSINESS_LAW_IDS: frozenset[str] = frozenset({
    "59/2020/qh14",   # Luật Doanh nghiệp 2020
    "61/2020/qh14",   # Luật Đầu tư 2020
    "67/2014/qh13",   # Luật Đầu tư 2014 (cũ)
    "68/2014/qh13",   # Luật Doanh nghiệp 2014 (cũ)
    "03/2016/qh14",   # Luật sửa đổi (đầu tư)
})

LAW_ID_TO_DOMAIN: dict[str, Domain] = (
    {law_id: "labor" for law_id in LABOR_LAW_IDS}
    | {law_id: "criminal" for law_id in CRIMINAL_LAW_IDS}
    | {law_id: "business" for law_id in BUSINESS_LAW_IDS}
)


# ============================================================
# Tier 2: Keyword regex per domain
# ============================================================
# Strategy: dùng `\b` boundary + keywords đặc thù domain để giảm false positive.
# `re.IGNORECASE` vì văn bản có thể có CHỮ HOA ở title.

DOMAIN_PATTERNS: dict[Domain, re.Pattern[str]] = {
    "labor": re.compile(
        r"(?:người lao động|hợp đồng lao động|bộ luật lao động|"
        r"bảo hiểm xã hội|bảo hiểm thất nghiệp|công đoàn|"
        r"tiền lương|tiền công|làm thêm giờ|chấm dứt hợp đồng lao động|"
        r"thai sản|nghỉ phép|sa thải|kỷ luật lao động|"
        r"an toàn (?:lao động|vệ sinh)|tai nạn lao động)",
        re.IGNORECASE,
    ),
    "criminal": re.compile(
        r"(?:bộ luật hình sự|tội phạm|tội (?:giết|trộm|cướp|lừa đảo|"
        r"đánh bạc|tham ô|nhận hối lộ|đưa hối lộ|cố ý gây thương tích|"
        r"hiếp dâm|cướp giật|buôn lậu|sản xuất.*ma túy|tàng trữ.*ma túy)|"
        r"hình phạt (?:tù|tiền|cải tạo|cảnh cáo|tử hình)|"
        r"truy cứu trách nhiệm hình sự|"
        r"phạt tù từ \d+|"
        r"cải tạo không giam giữ)",
        re.IGNORECASE,
    ),
    "business": re.compile(
        r"(?:doanh nghiệp|đầu tư|kinh doanh|đăng ký doanh nghiệp|"
        r"đăng ký kinh doanh|công ty (?:cổ phần|tnhh|hợp danh|tư nhân)|"
        r"vốn điều lệ|cổ đông|thành viên góp vốn|hộ kinh doanh|"
        r"giấy chứng nhận đăng ký|ngành nghề kinh doanh|"
        r"tổ chức kinh tế|nhà đầu tư|dự án đầu tư|"
        r"giải thể|phá sản|chia.*tách.*doanh nghiệp|"
        r"sáp nhập doanh nghiệp)",
        re.IGNORECASE,
    ),
}


@dataclass(slots=True)
class DomainMatch:
    """Kết quả classify domain cho 1 article."""

    domain: Domain | None
    score: int  # tổng số keyword matches
    via_law_id: bool  # True nếu match qua whitelist law_id (high confidence)
    matches_per_domain: dict[str, int]


def classify_domain(
    *,
    law_id: str = "",
    text: str = "",
    scan_chars: int = 1500,
    min_keyword_score: int = 2,
) -> DomainMatch:
    """Classify article về 1 trong 3 domains, hoặc None.

    Args:
        law_id: ID văn bản (đã lowercased).
        text: full text article (sẽ scan first ``scan_chars``).
        scan_chars: max chars to scan cho keyword matching.
        min_keyword_score: min score để consider a domain (giảm false positive).

    Returns:
        DomainMatch với domain (Optional) + score + via_law_id flag.
    """
    law_id_norm = law_id.strip().lower()

    # Tier 1: whitelist law_id (highest priority)
    if law_id_norm in LAW_ID_TO_DOMAIN:
        domain = LAW_ID_TO_DOMAIN[law_id_norm]
        return DomainMatch(
            domain=domain,
            score=999,
            via_law_id=True,
            matches_per_domain={domain: 999},
        )

    # Tier 2: keyword scan
    snippet = text[:scan_chars]
    matches: dict[str, int] = {
        domain: len(pattern.findall(snippet)) for domain, pattern in DOMAIN_PATTERNS.items()
    }

    best_domain = max(matches, key=lambda d: matches[d])
    best_score = matches[best_domain]

    if best_score < min_keyword_score:
        return DomainMatch(domain=None, score=best_score, via_law_id=False, matches_per_domain=matches)

    return DomainMatch(
        domain=best_domain,  # type: ignore[arg-type]
        score=best_score,
        via_law_id=False,
        matches_per_domain=matches,
    )

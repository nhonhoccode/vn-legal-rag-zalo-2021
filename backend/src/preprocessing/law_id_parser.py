"""Parse `law_id` của Zalo Legal 2021 → metadata (loại văn bản, cơ quan ban hành).

Format của `law_id` từ corpus: ``{number}/{year}/{type-issuer}``
Ví dụ: ``"47/2011/tt-bca"`` → 47, 2011, Thông tư - Bộ Công an
       ``"100/2015/qh13"``  → 100, 2015, (Bộ) Luật - Quốc hội khoá 13
       ``"45/2019/qh14"``   → 45, 2019, (Bộ) Luật - Quốc hội khoá 14
       ``"01/2009/tt-bnn"`` → 1, 2009, Thông tư - Bộ Nông nghiệp

Không cover hết tất cả issuer codes (Việt Nam có hàng chục bộ ngành) —
chỉ map những cái phổ biến + fallback "Cơ quan khác".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Loại văn bản.
TYPE_MAP: dict[str, str] = {
    "qh": "Luật/Bộ luật",  # Quốc hội ban hành
    "nđ": "Nghị định",
    "nd": "Nghị định",
    "tt": "Thông tư",
    "ttlt": "Thông tư liên tịch",
    "qđ": "Quyết định",
    "qd": "Quyết định",
    "ct": "Chỉ thị",
    "lệnh": "Lệnh",
    "lenh": "Lệnh",
    "nq": "Nghị quyết",
    "pl": "Pháp lệnh",
    "hđ": "Hiến định",
    "vbhn": "Văn bản hợp nhất",
}

# Cơ quan ban hành (subset phổ biến).
ISSUER_MAP: dict[str, str] = {
    "qh": "Quốc hội",
    "ubtvqh": "Ủy ban Thường vụ Quốc hội",
    "cp": "Chính phủ",
    "ttg": "Thủ tướng Chính phủ",
    "ctn": "Chủ tịch nước",
    # Bộ
    "bca": "Bộ Công an",
    "bqp": "Bộ Quốc phòng",
    "btc": "Bộ Tài chính",
    "bgtvt": "Bộ Giao thông vận tải",
    "byt": "Bộ Y tế",
    "bgddt": "Bộ Giáo dục và Đào tạo",
    "btttt": "Bộ Thông tin và Truyền thông",
    "blđtbxh": "Bộ Lao động Thương binh Xã hội",
    "bldtbxh": "Bộ Lao động Thương binh Xã hội",
    "btnmt": "Bộ Tài nguyên và Môi trường",
    "bxd": "Bộ Xây dựng",
    "bnv": "Bộ Nội vụ",
    "bnn": "Bộ Nông nghiệp và Phát triển nông thôn",
    "bnnptnt": "Bộ Nông nghiệp và Phát triển nông thôn",
    "bkhcn": "Bộ Khoa học và Công nghệ",
    "bvhttdl": "Bộ Văn hoá Thể thao và Du lịch",
    "bct": "Bộ Công Thương",
    "bkhđt": "Bộ Kế hoạch và Đầu tư",
    "bkhdt": "Bộ Kế hoạch và Đầu tư",
    "btp": "Bộ Tư pháp",
    # Tổ chức khác
    "nhnn": "Ngân hàng Nhà nước",
    "tandtc": "Toà án Nhân dân Tối cao",
    "vksndtc": "Viện Kiểm sát Nhân dân Tối cao",
    "ktnn": "Kiểm toán Nhà nước",
    "btnn": "Bộ Tài nguyên Nông nghiệp",
}


# QH + 2 chữ số khóa: "qh13", "qh14" → loại Luật/Bộ luật, issuer Quốc hội khóa N.
_QH_KHOA_RE = re.compile(r"^qh(\d{1,2})$")

# Format chuẩn: "<number>/<year>/<type-suffix>"
_LAW_ID_RE = re.compile(
    r"^(?P<number>\d+)/(?P<year>\d{4})/(?P<suffix>.+)$",
)


@dataclass(slots=True)
class LawMetadata:
    """Metadata derived từ law_id."""

    law_id: str
    number: str = ""
    year: str = ""
    type_code: str = ""
    issuer_code: str = ""
    type_full: str = ""
    issuer_full: str = ""
    raw_suffix: str = ""

    def synthetic_title(self) -> str:
        """Tạo title tổng hợp khi corpus không có sẵn law_title.

        Ví dụ: "Luật/Bộ luật số 45/2019 (Quốc hội khóa 14)"
               "Thông tư số 47/2011 (Bộ Công an)"
        """
        parts: list[str] = []
        if self.type_full:
            parts.append(self.type_full)
        parts.append(f"số {self.number}/{self.year}")
        if self.issuer_full:
            parts.append(f"({self.issuer_full})")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "law_id": self.law_id,
            "number": self.number,
            "year": self.year,
            "type_code": self.type_code,
            "issuer_code": self.issuer_code,
            "type_full": self.type_full,
            "issuer_full": self.issuer_full,
        }


def parse_law_id(law_id: str) -> LawMetadata:
    """Parse law_id format ``{number}/{year}/{type-issuer}`` → LawMetadata.

    Nếu không match format kỳ vọng → trả về metadata với fields rỗng (no crash).
    """
    if not law_id:
        return LawMetadata(law_id="")

    law_id = law_id.strip().lower()
    md = LawMetadata(law_id=law_id)

    m = _LAW_ID_RE.match(law_id)
    if not m:
        return md

    md.number = m.group("number")
    md.year = m.group("year")
    md.raw_suffix = m.group("suffix")

    suffix = md.raw_suffix

    # Case 1: "qhNN" — Luật/Bộ luật
    qh_match = _QH_KHOA_RE.match(suffix)
    if qh_match:
        khoa = qh_match.group(1)
        md.type_code = "qh"
        md.type_full = TYPE_MAP["qh"]
        md.issuer_code = "qh"
        md.issuer_full = f"Quốc hội khóa {khoa}"
        return md

    # Case 2: "ubtvqhNN" — UBTVQH
    if suffix.startswith("ubtvqh"):
        md.type_code = "nq"
        md.type_full = TYPE_MAP.get("nq", "")
        md.issuer_code = "ubtvqh"
        md.issuer_full = "Ủy ban Thường vụ Quốc hội"
        return md

    # Case 3: "<type>-<issuer>" — split bằng dấu "-"
    if "-" in suffix:
        parts = suffix.split("-", 1)
        md.type_code = parts[0]
        md.issuer_code = parts[1]
        md.type_full = TYPE_MAP.get(md.type_code, "")
        md.issuer_full = ISSUER_MAP.get(md.issuer_code, "")
        return md

    # Case 4: chỉ có 1 token (không có dash) → coi là issuer_code
    md.issuer_code = suffix
    md.issuer_full = ISSUER_MAP.get(suffix, "")
    return md

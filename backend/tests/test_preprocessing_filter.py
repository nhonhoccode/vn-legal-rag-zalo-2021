"""Tests cho filter_domain — 3 ngành Decision 20."""

from src.preprocessing.filter_domain import (
    BUSINESS_LAW_IDS,
    CRIMINAL_LAW_IDS,
    LABOR_LAW_IDS,
    classify_domain,
)


# ============================================================
# Tier 1: whitelist law_id
# ============================================================
def test_classify_via_law_id_labor():
    m = classify_domain(law_id="45/2019/qh14", text="abc")
    assert m.domain == "labor"
    assert m.via_law_id is True
    assert m.score == 999


def test_classify_via_law_id_criminal():
    m = classify_domain(law_id="100/2015/qh13", text="abc")
    assert m.domain == "criminal"
    assert m.via_law_id is True


def test_classify_via_law_id_business_dn():
    m = classify_domain(law_id="59/2020/qh14", text="abc")
    assert m.domain == "business"
    assert m.via_law_id is True


def test_classify_via_law_id_business_dt():
    m = classify_domain(law_id="61/2020/qh14", text="abc")
    assert m.domain == "business"


def test_classify_via_law_id_case_insensitive():
    m = classify_domain(law_id="45/2019/QH14", text="abc")
    assert m.domain == "labor"


# ============================================================
# Tier 2: keyword scan
# ============================================================
def test_classify_via_keyword_labor():
    text = (
        "Người sử dụng lao động phải bồi thường cho người lao động khi đơn phương "
        "chấm dứt hợp đồng lao động trái pháp luật. Bảo hiểm xã hội phải đóng đủ."
    )
    m = classify_domain(law_id="999/2020/tt-blđtbxh", text=text)
    assert m.domain == "labor"
    assert m.via_law_id is False
    assert m.score >= 2


def test_classify_via_keyword_criminal():
    text = (
        "Người nào trộm cắp tài sản trị giá trên 2 triệu đồng thì bị truy cứu trách nhiệm "
        "hình sự. Phạt tù từ 06 tháng đến 03 năm."
    )
    m = classify_domain(law_id="999/2020/nđ-cp", text=text)
    assert m.domain == "criminal"
    assert m.score >= 2


def test_classify_via_keyword_business():
    text = (
        "Doanh nghiệp tư nhân phải đăng ký doanh nghiệp tại cơ quan đăng ký kinh doanh. "
        "Vốn điều lệ tối thiểu là 100 triệu đồng. Cổ đông phải thực hiện nghĩa vụ góp vốn."
    )
    m = classify_domain(law_id="999/2020/nd-cp", text=text)
    assert m.domain == "business"


def test_classify_no_match_returns_none():
    text = "Quy định về bảo vệ môi trường biển và hệ sinh thái."
    m = classify_domain(law_id="999/2020/tt-btnmt", text=text)
    assert m.domain is None


def test_classify_below_threshold_returns_none():
    """1 keyword đơn lẻ → không đủ confidence."""
    text = "Quy định về tiền lương tối thiểu."  # chỉ 1 match labor
    m = classify_domain(law_id="999/2020/tt-blđtbxh", text=text, min_keyword_score=2)
    assert m.domain is None


def test_classify_picks_max_when_multi_domain():
    """Text mention nhiều domain → pick best score."""
    text = (
        "Doanh nghiệp đầu tư phải đăng ký kinh doanh. Doanh nghiệp nộp thuế đúng hạn. "
        "Doanh nghiệp có quyền thuê lao động và ký hợp đồng lao động."
    )
    m = classify_domain(law_id="999/2020/cp", text=text)
    # business mentions > labor mentions
    assert m.domain == "business"


def test_law_id_priority_over_keyword():
    """Whitelist law_id luôn thắng kể cả keyword nói khác."""
    m = classify_domain(
        law_id="45/2019/qh14",  # BLLĐ — labor
        text="Tội trộm cắp tài sản. Hình phạt tù.",
    )
    assert m.domain == "labor"
    assert m.via_law_id is True


# ============================================================
# Coverage of whitelisted law_ids
# ============================================================
def test_labor_law_ids_non_empty():
    assert len(LABOR_LAW_IDS) >= 2  # ít nhất BLLĐ 2012 + 2019


def test_criminal_law_ids_non_empty():
    assert len(CRIMINAL_LAW_IDS) >= 1


def test_business_law_ids_non_empty():
    assert len(BUSINESS_LAW_IDS) >= 2  # Luật DN + Luật ĐT


def test_no_overlap_between_domains():
    """Một law_id không thể thuộc 2 ngành cùng lúc."""
    assert not (LABOR_LAW_IDS & CRIMINAL_LAW_IDS)
    assert not (LABOR_LAW_IDS & BUSINESS_LAW_IDS)
    assert not (CRIMINAL_LAW_IDS & BUSINESS_LAW_IDS)

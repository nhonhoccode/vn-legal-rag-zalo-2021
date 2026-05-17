"""Tests cho law_id_parser."""

from src.preprocessing.law_id_parser import LawMetadata, parse_law_id


def test_parse_qh14_law():
    """45/2019/qh14 = Bộ luật Lao động."""
    md = parse_law_id("45/2019/qh14")
    assert md.number == "45"
    assert md.year == "2019"
    assert md.type_code == "qh"
    assert md.type_full == "Luật/Bộ luật"
    assert md.issuer_code == "qh"
    assert md.issuer_full == "Quốc hội khóa 14"


def test_parse_qh13():
    md = parse_law_id("100/2015/qh13")
    assert md.issuer_full == "Quốc hội khóa 13"


def test_parse_thong_tu_bca():
    md = parse_law_id("47/2011/tt-bca")
    assert md.number == "47"
    assert md.year == "2011"
    assert md.type_code == "tt"
    assert md.type_full == "Thông tư"
    assert md.issuer_code == "bca"
    assert md.issuer_full == "Bộ Công an"


def test_parse_thong_tu_bnn():
    md = parse_law_id("01/2009/tt-bnn")
    assert md.issuer_code == "bnn"
    assert md.issuer_full == "Bộ Nông nghiệp và Phát triển nông thôn"


def test_parse_nghi_dinh_cp():
    md = parse_law_id("100/2019/nđ-cp")
    assert md.type_full == "Nghị định"
    assert md.issuer_full == "Chính phủ"


def test_parse_nghi_dinh_no_diacritic():
    md = parse_law_id("100/2019/nd-cp")
    assert md.type_full == "Nghị định"


def test_parse_unknown_issuer_returns_empty_full():
    md = parse_law_id("1/2020/tt-xyz")
    assert md.type_full == "Thông tư"
    assert md.issuer_code == "xyz"
    assert md.issuer_full == ""  # unknown → empty


def test_parse_invalid_format_no_crash():
    md = parse_law_id("not-a-law-id")
    assert md.law_id == "not-a-law-id"
    assert md.number == ""
    assert md.year == ""


def test_parse_empty_string():
    md = parse_law_id("")
    assert isinstance(md, LawMetadata)
    assert md.law_id == ""


def test_synthetic_title_for_qh():
    md = parse_law_id("45/2019/qh14")
    title = md.synthetic_title()
    assert "Luật/Bộ luật" in title
    assert "45/2019" in title
    assert "Quốc hội khóa 14" in title


def test_synthetic_title_for_tt():
    md = parse_law_id("47/2011/tt-bca")
    title = md.synthetic_title()
    assert "Thông tư" in title
    assert "47/2011" in title
    assert "Bộ Công an" in title


def test_lowercase_normalization():
    md = parse_law_id("45/2019/QH14")  # uppercase
    assert md.law_id == "45/2019/qh14"
    assert md.issuer_full == "Quốc hội khóa 14"


def test_to_dict_round_trip():
    md = parse_law_id("47/2011/tt-bca")
    d = md.to_dict()
    assert d["law_id"] == "47/2011/tt-bca"
    assert d["type_full"] == "Thông tư"
    assert d["issuer_full"] == "Bộ Công an"

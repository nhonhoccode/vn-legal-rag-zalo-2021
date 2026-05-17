"""Tests cho `src.ingestion.base` — Article, GoldenQA dataclasses."""

from src.ingestion.base import Article, GoldenQA


def test_article_to_dict_round_trip():
    a = Article(
        law_id="45/2019/QH14",
        law_title="Bộ luật Lao động",
        article_id="41",
        text="Điều 41. ...",
        extra={"category": "labor"},
    )
    d = a.to_dict()
    assert d["law_id"] == "45/2019/QH14"
    assert d["law_title"] == "Bộ luật Lao động"
    assert d["article_id"] == "41"
    assert d["text"].startswith("Điều 41")
    assert d["extra"]["category"] == "labor"


def test_article_extra_default_empty_dict():
    a = Article(law_id="L1", law_title="t", article_id="1", text="x")
    assert a.extra == {}


def test_golden_qa_to_dict():
    qa = GoldenQA(
        question_id="q_001",
        question="Lao động bị sa thải có được bồi thường?",
        relevant_articles=[{"law_id": "45/2019/QH14", "article_id": "41"}],
    )
    d = qa.to_dict()
    assert d["question_id"] == "q_001"
    assert len(d["relevant_articles"]) == 1
    assert d["relevant_articles"][0]["law_id"] == "45/2019/QH14"


def test_article_slots():
    """Verify slots=True works (no __dict__)."""
    a = Article(law_id="L1", law_title="t", article_id="1", text="x")
    # slots=True objects don't have __dict__
    assert not hasattr(a, "__dict__")

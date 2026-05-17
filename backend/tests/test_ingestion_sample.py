"""Tests cho `src.ingestion.sample_data` + JSONL save helpers."""

import json

from src.ingestion.base import Article, GoldenQA
from src.ingestion.sample_data import SampleDataSource
from src.ingestion.zalo_legal import save_articles_to_jsonl, save_qa_to_jsonl


def test_sample_source_name():
    assert SampleDataSource().name == "sample"


def test_sample_corpus_yields_articles():
    articles = list(SampleDataSource().fetch_corpus())
    assert len(articles) >= 5
    assert all(isinstance(a, Article) for a in articles)


def test_sample_corpus_covers_3_categories():
    """Per Decision 20: focus 3 ngành Lao động + Doanh nghiệp/Đầu tư + Hình sự."""
    categories = {a.extra.get("category") for a in SampleDataSource().fetch_corpus()}
    # business covers cả Doanh nghiệp + Đầu tư
    assert {"labor", "criminal", "business"}.issubset(categories)


def test_sample_articles_have_complete_metadata():
    for art in SampleDataSource().fetch_corpus():
        assert art.law_id, "law_id required"
        assert art.law_title, "law_title required"
        assert art.article_id, "article_id required"
        assert art.text, "text required"
        assert "Điều" in art.text, "text should contain 'Điều' header"


def test_sample_qa_yields_items():
    qa = list(SampleDataSource().fetch_qa())
    assert len(qa) >= 3
    assert all(isinstance(q, GoldenQA) for q in qa)


def test_sample_qa_relevant_articles_exist_in_corpus():
    """Critical: golden Q&A phải point đến articles thực sự có trong corpus."""
    src = SampleDataSource()
    article_keys = {(a.law_id, a.article_id) for a in src.fetch_corpus()}
    for qa in src.fetch_qa():
        for relevant in qa.relevant_articles:
            key = (relevant["law_id"], relevant["article_id"])
            assert key in article_keys, (
                f"QA {qa.question_id} references missing article {key}; "
                f"available: {article_keys}"
            )


def test_save_articles_jsonl_round_trip(tmp_path):
    out = tmp_path / "corpus.jsonl"
    n = save_articles_to_jsonl(SampleDataSource().fetch_corpus(), out)

    assert n > 0
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == n

    first = records[0]
    assert {"law_id", "law_title", "article_id", "text", "extra"} <= first.keys()


def test_save_qa_jsonl_round_trip(tmp_path):
    out = tmp_path / "qa.jsonl"
    n = save_qa_to_jsonl(SampleDataSource().fetch_qa(), out)

    assert n > 0
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == n

    first = records[0]
    assert "question" in first
    assert "relevant_articles" in first


def test_jsonl_preserves_vietnamese_characters(tmp_path):
    """Tiếng Việt phải lưu raw UTF-8, KHÔNG escape \\uXXXX."""
    out = tmp_path / "corpus.jsonl"
    save_articles_to_jsonl(SampleDataSource().fetch_corpus(), out)

    raw = out.read_text(encoding="utf-8")
    assert "Điều" in raw
    assert "Bộ luật" in raw
    assert "\\u" not in raw, "Vietnamese should be UTF-8, not escaped"


def test_save_creates_parent_dirs(tmp_path):
    out = tmp_path / "deeply" / "nested" / "corpus.jsonl"
    n = save_articles_to_jsonl(SampleDataSource().fetch_corpus(), out)
    assert out.exists()
    assert n > 0

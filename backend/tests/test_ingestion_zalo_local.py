"""Tests cho `ZaloLegalLocalSource` — load từ file local."""

import json

import pytest

from src.ingestion.zalo_legal import ZaloLegalLocalSource


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_local_source_jsonl_nested_schema(tmp_path):
    """Schema A: {law_id, articles: [...]}"""
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [
        {
            "law_id": "L1",
            "law_title": "Luật A",
            "articles": [
                {"article_id": "1", "text": "Điều 1: nội dung A"},
                {"article_id": "2", "text": "Điều 2: nội dung B"},
            ],
        },
        {
            "law_id": "L2",
            "law_title": "Luật B",
            "articles": [{"article_id": "1", "text": "Điều 1: nội dung C"}],
        },
    ])

    src = ZaloLegalLocalSource(corpus_path=corpus)
    arts = list(src.fetch_corpus())
    assert len(arts) == 3
    assert arts[0].law_id == "L1"
    assert arts[0].article_id == "1"
    assert arts[2].law_id == "L2"


def test_local_source_json_array(tmp_path):
    corpus = tmp_path / "corpus.json"
    _write_json(corpus, [
        {"law_id": "L1", "law_title": "Luật A", "articles": [
            {"article_id": "1", "text": "x"}]},
    ])
    src = ZaloLegalLocalSource(corpus_path=corpus)
    arts = list(src.fetch_corpus())
    assert len(arts) == 1


def test_local_source_flat_schema(tmp_path):
    """Schema B: flat {law_id, article_id, text}"""
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [
        {"law_id": "L1", "article_id": "1", "text": "Điều 1"},
        {"law_id": "L1", "article_id": "2", "text": "Điều 2"},
    ])
    src = ZaloLegalLocalSource(corpus_path=corpus)
    arts = list(src.fetch_corpus())
    assert len(arts) == 2
    assert arts[0].article_id == "1"


def test_local_source_missing_corpus_raises(tmp_path):
    src = ZaloLegalLocalSource(corpus_path=tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        list(src.fetch_corpus())


def test_local_source_qa_optional(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"law_id": "L1", "articles": []}])
    src = ZaloLegalLocalSource(corpus_path=corpus)
    assert list(src.fetch_qa()) == []


def test_local_source_qa_jsonl(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"law_id": "L1", "articles": []}])
    qa = tmp_path / "qa.jsonl"
    _write_jsonl(qa, [
        {
            "question_id": "q1",
            "question": "Câu hỏi 1?",
            "relevant_articles": [{"law_id": "L1", "article_id": "1"}],
        },
    ])
    src = ZaloLegalLocalSource(corpus_path=corpus, qa_path=qa)
    items = list(src.fetch_qa())
    assert len(items) == 1
    assert items[0].question == "Câu hỏi 1?"


def test_local_source_unsupported_extension_raises(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("foo", encoding="utf-8")
    src = ZaloLegalLocalSource(corpus_path=corpus)
    with pytest.raises(ValueError, match="Unsupported file extension"):
        list(src.fetch_corpus())


def test_local_source_skips_unknown_record_schema(tmp_path, caplog):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [
        {"law_id": "L1", "articles": [{"article_id": "1", "text": "ok"}]},
        {"foo": "bar"},  # unknown shape — should be skipped with warning
    ])
    src = ZaloLegalLocalSource(corpus_path=corpus)
    arts = list(src.fetch_corpus())
    assert len(arts) == 1  # 2nd record skipped

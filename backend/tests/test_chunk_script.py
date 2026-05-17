"""End-to-end test cho `scripts/03_chunk.py`."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _import():
    spec = importlib.util.spec_from_file_location("chunk_script", SCRIPTS_DIR / "03_chunk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_chunk_script_processes_articles(tmp_path):
    inp = tmp_path / "articles.jsonl"
    _write_jsonl(inp, [
        {
            "law_id": "45/2019/qh14",
            "law_title": "Bộ luật Lao động",
            "article_id": "1",
            "text": "Điều 1. Phạm vi điều chỉnh\nQuy định tiêu chuẩn lao động.",
            "domain": "labor",
            "metadata": {"type_full": "Luật/Bộ luật"},
        },
        {
            "law_id": "100/2015/qh13",
            "law_title": "Bộ luật Hình sự",
            "article_id": "8",
            "text": "Điều 8. Khái niệm tội phạm\nTội phạm là hành vi nguy hiểm cho xã hội.",
            "domain": "criminal",
            "metadata": {},
        },
    ])
    out = tmp_path / "chunks.jsonl"
    script = _import()
    rc = script.main(["--input", str(inp), "--output", str(out)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 2
    chunk_ids = {r["chunk_id"] for r in records}
    assert len(chunk_ids) == 2  # unique


def test_chunk_script_drops_too_short(tmp_path):
    inp = tmp_path / "a.jsonl"
    _write_jsonl(inp, [
        {"law_id": "L1", "article_id": "1", "text": "ngắn", "domain": "labor", "metadata": {}}
    ])
    out = tmp_path / "c.jsonl"
    script = _import()
    rc = script.main(["--input", str(inp), "--output", str(out)])
    assert rc == 1  # 0 chunks


def test_chunk_script_missing_input(tmp_path):
    script = _import()
    rc = script.main(["--input", str(tmp_path / "nope.jsonl"), "--output", str(tmp_path / "o.jsonl")])
    assert rc == 1


def test_chunk_script_limit(tmp_path):
    inp = tmp_path / "a.jsonl"
    _write_jsonl(inp, [
        {"law_id": "L1", "article_id": str(i),
         "text": "Điều X. Nội dung article đủ dài qua threshold min chars.",
         "domain": "labor", "metadata": {}}
        for i in range(10)
    ])
    out = tmp_path / "c.jsonl"
    script = _import()
    rc = script.main(["--input", str(inp), "--output", str(out), "--limit", "3"])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 3

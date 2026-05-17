"""End-to-end test cho `scripts/02_preprocess.py`."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _import_script():
    spec = importlib.util.spec_from_file_location(
        "preprocess_script", SCRIPTS_DIR / "02_preprocess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_script_processes_real_zalo_schema(tmp_path):
    """Real Zalo Legal 2021 corpus có schema flat: {law_id, article_id, title, text}."""
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [
        {
            "law_id": "45/2019/qh14",  # BLLĐ — sẽ pass via law_id
            "law_title": "",
            "article_id": "45/2019/qh14__41",
            "text": "Điều 41. Bồi thường khi chấm dứt hợp đồng lao động trái pháp luật\n"
                    "Người sử dụng lao động phải bồi thường ít nhất 02 tháng tiền lương "
                    "cho người lao động.",
            "extra": {},
        },
        {
            "law_id": "999/2020/tt-btnmt",  # môi trường — sẽ bị filter
            "article_id": "1",
            "text": "Quy định về bảo vệ môi trường biển và hệ sinh thái rạn san hô.",
        },
        {
            "law_id": "100/2015/qh13",  # BLHS — sẽ pass
            "article_id": "173",
            "text": "Điều 173. Tội trộm cắp tài sản\nNgười nào trộm cắp tài sản bị phạt tù.",
        },
    ])

    out = tmp_path / "out.jsonl"
    script = _import_script()
    rc = script.main(["--input", str(raw), "--output", str(out)])
    assert rc == 0
    assert out.exists()

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    # 1st (BLLĐ) + 3rd (BLHS) kept; 2nd (môi trường) filtered
    assert len(records) == 2

    domains = {r["domain"] for r in records}
    assert domains == {"labor", "criminal"}

    # Verify enrichment
    bllt = next(r for r in records if r["domain"] == "labor")
    assert bllt["metadata"]["type_full"] == "Luật/Bộ luật"
    assert bllt["metadata"]["issuer_full"] == "Quốc hội khóa 14"
    assert bllt["metadata"]["domain_via"] == "law_id"


def test_script_no_filter_keeps_all(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [
        {"law_id": "999/2020/tt-btnmt", "article_id": "1",
         "text": "Quy định về môi trường biển dài hơn 20 ký tự để không bị too short."},
    ])
    out = tmp_path / "out.jsonl"
    script = _import_script()
    rc = script.main(["--input", str(raw), "--output", str(out), "--no-filter"])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    assert records[0]["domain"] == "unknown"


def test_script_drops_too_short(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [
        {"law_id": "45/2019/qh14", "article_id": "1", "text": "ngắn"},  # < 20 chars
    ])
    out = tmp_path / "out.jsonl"
    script = _import_script()
    rc = script.main(["--input", str(raw), "--output", str(out)])
    # 0 articles → return 1
    assert rc == 1


def test_script_missing_input_returns_error(tmp_path):
    script = _import_script()
    rc = script.main(["--input", str(tmp_path / "nope.jsonl"), "--output", str(tmp_path / "o.jsonl")])
    assert rc == 1


def test_script_limit_works(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [
        {"law_id": "45/2019/qh14", "article_id": str(i),
         "text": "Người lao động được bồi thường khi đơn phương chấm dứt hợp đồng lao động."}
        for i in range(10)
    ])
    out = tmp_path / "out.jsonl"
    script = _import_script()
    rc = script.main(["--input", str(raw), "--output", str(out), "--limit", "3"])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 3


def test_script_unicode_normalized_in_output(tmp_path):
    """Output text phải NFC normalized."""
    raw = tmp_path / "raw.jsonl"
    # "ò" với 2 codepoints (NFD form)
    decomposed = "ò"  # noqa: RUF001
    text = f"Điều 1. {decomposed} là tiếng Việt. Người lao động được bồi thường lương."
    _write_jsonl(raw, [
        {"law_id": "45/2019/qh14", "article_id": "1", "text": text},
    ])
    out = tmp_path / "out.jsonl"
    script = _import_script()
    rc = script.main(["--input", str(raw), "--output", str(out)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    # NFC: "ò" should be 1 codepoint
    out_text = records[0]["text"]
    import unicodedata
    assert unicodedata.normalize("NFC", out_text) == out_text

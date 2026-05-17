"""End-to-end test cho `scripts/01_pull_dataset.py` chạy với --source sample."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts dir to path so we can import 01_pull_dataset
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _import_script_module():
    """Import `01_pull_dataset` (filename starts with digit, can't use normal import)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pull_dataset_script",
        SCRIPTS_DIR / "01_pull_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_sample_mode_creates_files(tmp_path):
    """Run script với --source sample → tạo corpus.jsonl + qa.jsonl."""
    script = _import_script_module()
    rc = script.main(
        ["--source", "sample", "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 0

    out_dir = tmp_path / "out"
    corpus = out_dir / "corpus.jsonl"
    qa = out_dir / "qa.jsonl"

    assert corpus.exists()
    assert qa.exists()

    # Content sanity check
    corpus_records = [
        json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line
    ]
    qa_records = [
        json.loads(line) for line in qa.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(corpus_records) >= 5
    assert len(qa_records) >= 3


def test_script_argparse_local_requires_corpus():
    """--source local without --corpus → SystemExit."""
    script = _import_script_module()
    with pytest.raises(SystemExit):
        script.main(["--source", "local"])


def test_script_choice_validation():
    script = _import_script_module()
    # invalid source
    with pytest.raises(SystemExit):
        script.main(["--source", "invalid"])


def test_script_returns_nonzero_for_empty_corpus(tmp_path, monkeypatch):
    """Edge case: nếu source yield 0 articles → script return 1."""
    script = _import_script_module()

    class _EmptySource:
        @property
        def name(self) -> str:
            return "empty"

        def fetch_corpus(self):
            return iter([])

        def fetch_qa(self):
            return iter([])

    monkeypatch.setattr(script, "get_source", lambda _args: _EmptySource())
    rc = script.main(["--source", "sample", "--output-dir", str(tmp_path / "out")])
    assert rc == 1


@pytest.mark.integration
def test_script_hf_mode_real_download(tmp_path):
    """Integration test: cần internet + HF dataset thật.

    Chạy với: pytest -m integration
    Mặc định bị skip để CI nhanh + không phụ thuộc network.
    """
    script = _import_script_module()
    # Sẽ thử KNOWN_HF_DATASETS — nếu tất cả fail, RuntimeError
    rc = script.main(
        ["--source", "hf", "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 0
    assert (tmp_path / "out" / "corpus.jsonl").exists()

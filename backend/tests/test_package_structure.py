"""Smoke test: verify package structure được tạo đúng theo ARCHITECTURE.md."""

import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"

EXPECTED_PACKAGES = [
    "src",
    "src.ingestion",
    "src.preprocessing",
    "src.chunking",
    "src.embeddings",
    "src.vectorstore",
    "src.retrieval",
    "src.reranking",
    "src.generation",
    "src.cache",
    "src.api",
    "src.api.middleware",
    "src.api.routes",
    "src.eval",
]


def test_all_packages_have_init():
    for pkg in EXPECTED_PACKAGES:
        path = BACKEND_SRC.parent / Path(*pkg.split("."))
        assert path.is_dir(), f"Missing dir: {path}"
        init_file = path / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py: {init_file}"


def test_all_packages_importable():
    for pkg in EXPECTED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            raise AssertionError(f"Cannot import {pkg}: {e}")  # noqa: B904


def test_data_directories_exist():
    expected = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "eval",
        PROJECT_ROOT / "data" / "eval" / "results",
        PROJECT_ROOT / "data" / "chroma_db",
    ]
    for d in expected:
        assert d.is_dir(), f"Missing data directory: {d}"


def test_docs_directory_intact():
    docs = PROJECT_ROOT / "docs"
    expected_docs = [
        "PROJECT_OVERVIEW.md",
        "TECH_STACK.md",
        "ARCHITECTURE.md",
        "DATA_STRATEGY.md",
        "RETRIEVAL_STRATEGY.md",
        "EVALUATION_STRATEGY.md",
        "ROADMAP.md",
        "DECISIONS.md",
        "NEEDS_CLARIFICATION.md",
    ]
    for f in expected_docs:
        assert (docs / f).exists(), f"Missing doc: {f}"

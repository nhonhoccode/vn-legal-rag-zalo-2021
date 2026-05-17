"""Pull Zalo Legal 2021 dataset → `data/raw/zalo_legal/{corpus,qa}.jsonl`.

Usage:

    # Smoke test (always works, ~10 articles + 5 Q&A):
    python scripts/01_pull_dataset.py --source sample

    # Real download from HuggingFace community mirror:
    python scripts/01_pull_dataset.py --source hf
    python scripts/01_pull_dataset.py --source hf --dataset thinhlpg/zalo-legal-text-retrieval

    # Local files (user manually downloaded):
    python scripts/01_pull_dataset.py --source local \\
        --corpus /path/to/corpus.json --qa /path/to/qa.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add backend/ to path so we can `from src...`
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config import PROJECT_ROOT  # noqa: E402
from src.ingestion.base import IngestionSource  # noqa: E402
from src.ingestion.sample_data import SampleDataSource  # noqa: E402
from src.ingestion.zalo_legal import (  # noqa: E402
    ZaloLegalHFSource,
    ZaloLegalLocalSource,
    save_articles_to_jsonl,
    save_qa_to_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull Zalo Legal 2021 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source",
        choices=["sample", "hf", "local"],
        default="sample",
        help="Data source (default: sample)",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="(deprecated alias for --corpus-dataset)",
    )
    p.add_argument(
        "--corpus-dataset",
        default=None,
        help="HF corpus dataset name (only used if --source hf). Auto-pair với QA nếu nằm trong KNOWN_HF_DATASET_PAIRS.",
    )
    p.add_argument(
        "--qa-dataset",
        default=None,
        help="HF QA dataset name (only used if --source hf). Auto nếu trong known pair.",
    )
    p.add_argument(
        "--corpus",
        default=None,
        help="Local corpus path (only used if --source local).",
    )
    p.add_argument(
        "--qa",
        default=None,
        help="Local QA path (only used if --source local).",
    )
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "zalo_legal"),
        help="Output directory for corpus.jsonl + qa.jsonl",
    )
    return p.parse_args(argv)


def get_source(args: argparse.Namespace) -> IngestionSource:
    if args.source == "sample":
        return SampleDataSource()
    if args.source == "hf":
        return ZaloLegalHFSource(
            corpus_dataset=args.corpus_dataset or args.dataset,
            qa_dataset=args.qa_dataset,
        )
    if args.source == "local":
        if not args.corpus:
            raise SystemExit("--corpus is required when --source=local")
        return ZaloLegalLocalSource(
            corpus_path=Path(args.corpus),
            qa_path=Path(args.qa) if args.qa else None,
        )
    raise SystemExit(f"Unknown source: {args.source}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = get_source(args)
    output_dir = Path(args.output_dir)

    logger.info("Source: %s", source.name)
    logger.info("Output dir: %s", output_dir)

    corpus_path = output_dir / "corpus.jsonl"
    qa_path = output_dir / "qa.jsonl"

    # Corpus
    n_articles = save_articles_to_jsonl(source.fetch_corpus(), corpus_path)
    logger.info("Saved %d articles → %s", n_articles, corpus_path)

    # Q&A (optional — sample/local có; hf tùy schema)
    try:
        n_qa = save_qa_to_jsonl(source.fetch_qa(), qa_path)
        logger.info("Saved %d Q&A → %s", n_qa, qa_path)
    except (NotImplementedError, FileNotFoundError) as e:
        logger.warning("No Q&A available: %s", e)
        n_qa = 0

    logger.info("=" * 60)
    logger.info("DONE — Articles: %d  |  Q&A: %d", n_articles, n_qa)
    logger.info("=" * 60)

    if n_articles == 0:
        logger.error("No articles saved — abort.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

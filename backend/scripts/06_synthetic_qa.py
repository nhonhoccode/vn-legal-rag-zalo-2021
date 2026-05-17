"""Generate synthetic Q&A từ processed chunks.

Usage:
    python scripts/06_synthetic_qa.py --n 50
    python scripts/06_synthetic_qa.py --input data/processed/articles_clean.jsonl --n 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.api.dependencies import get_llm_client  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402
from src.eval.synthetic_qa import (  # noqa: E402
    generate_synthetic_qa_set,
    save_synthetic_qa_to_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic Q&A")
    p.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "articles_clean.jsonl"),
    )
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "synthetic_qa.jsonl"),
    )
    p.add_argument("--n", type=int, default=50, help="Số chunks sample")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


async def main_async(args) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s — run Phase 3 trước.", input_path)
        return 1

    # Load chunks
    chunks: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks", len(chunks))

    if not chunks:
        return 1

    llm = get_llm_client()
    logger.info("Using LLM: %s (%s)", llm.name, llm.model)
    logger.info("Generating synthetic Q&A for %d sampled chunks...", args.n)

    items = await generate_synthetic_qa_set(chunks, llm, n_samples=args.n, seed=args.seed)
    logger.info("Generated %d Q&A items total", len(items))

    output = Path(args.output)
    n_saved = save_synthetic_qa_to_jsonl(items, output)
    logger.info("Saved %d items → %s", n_saved, output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

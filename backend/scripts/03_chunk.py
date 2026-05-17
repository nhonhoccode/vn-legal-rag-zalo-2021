"""Chunk processed articles → chunks JSONL.

Pipeline:
    data/processed/articles_clean.jsonl  (Phase 3 output)
        ↓
    chunk_article (chunk-by-Điều, fallback Khoản/Điểm)
        ↓
    data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.chunking.legal_chunker import chunk_article  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk articles → chunks JSONL")
    p.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "articles_clean.jsonl"),
    )
    p.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
    )
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        logger.error("Hint: chạy `python scripts/02_preprocess.py` trước.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("max_tokens=%d  min_chars=%d", args.max_tokens, args.min_chars)

    n_articles = 0
    n_chunks = 0
    n_dup_skipped = 0
    seen_ids: set[str] = set()
    strategy_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    token_buckets: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            n_articles += 1

            for chunk in chunk_article(
                article=article,
                max_chunk_tokens=args.max_tokens,
                min_chunk_chars=args.min_chars,
            ):
                if chunk.chunk_id in seen_ids:
                    n_dup_skipped += 1
                    continue
                seen_ids.add(chunk.chunk_id)
                f_out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                n_chunks += 1
                strategy_counter[chunk.metadata.get("chunk_strategy", "?")] += 1
                domain_counter[chunk.domain] += 1
                tok = chunk.metadata.get("token_len", 0)
                bucket = (
                    "0-100"
                    if tok < 100
                    else "100-300"
                    if tok < 300
                    else "300-500"
                    if tok < 500
                    else "500+"
                )
                token_buckets[bucket] += 1

    logger.info("=" * 60)
    logger.info("Articles:                %d", n_articles)
    logger.info("Chunks:                  %d", n_chunks)
    if n_dup_skipped:
        logger.warning(
            "Duplicate chunk_ids skipped: %d (likely duplicate articles in input)",
            n_dup_skipped,
        )
    if n_articles:
        logger.info("Avg chunks/article:      %.2f", n_chunks / n_articles)
    logger.info("-" * 60)
    logger.info("Strategy:")
    for s, n in strategy_counter.most_common():
        logger.info("  %-30s %6d", s, n)
    logger.info("Token length buckets:")
    for b in ["0-100", "100-300", "300-500", "500+"]:
        if b in token_buckets:
            pct = 100 * token_buckets[b] / n_chunks
            logger.info("  %-10s %6d  (%5.1f%%)", b, token_buckets[b], pct)
    logger.info("Domain:")
    for d, n in domain_counter.most_common():
        logger.info("  %-12s %6d", d, n)
    logger.info("=" * 60)

    if n_chunks == 0:
        logger.error("No chunks produced — abort.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

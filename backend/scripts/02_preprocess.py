"""Preprocess raw corpus → cleaned + filtered (3 ngành) → JSONL.

Pipeline:
    raw corpus.jsonl
        ↓
    NFC normalize + remove zero-width + clean noise (unicode_norm + cleaning)
        ↓
    enrich metadata từ law_id (law_id_parser)
        ↓
    filter 3 ngành (filter_domain) — drop articles không match
        ↓
    drop too-short articles (< 20 chars)
        ↓
    data/processed/articles_clean.jsonl

Usage:

    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --input data/raw/zalo_legal/corpus.jsonl
    python scripts/02_preprocess.py --limit 1000  # smoke test
    python scripts/02_preprocess.py --no-filter  # giữ tất cả articles (skip 3-ngành filter)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

# Add backend/ to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config import PROJECT_ROOT  # noqa: E402
from src.preprocessing.cleaning import clean, is_too_short  # noqa: E402
from src.preprocessing.filter_domain import classify_domain  # noqa: E402
from src.preprocessing.law_id_parser import parse_law_id  # noqa: E402
from src.preprocessing.unicode_norm import full_normalize  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess raw legal corpus → cleaned + filtered JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "zalo_legal" / "corpus.jsonl"),
        help="Input raw corpus JSONL",
    )
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "articles_clean.jsonl"),
        help="Output processed JSONL",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit input records (for smoke test). None = all.",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip 3-ngành filter — giữ tất cả articles.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Drop articles với text ngắn hơn N chars (sau clean).",
    )
    return p.parse_args(argv)


def iter_input_records(input_path: Path, limit: int | None) -> Iterator[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def process_record(
    record: dict,
    *,
    apply_filter: bool,
    min_chars: int,
) -> tuple[dict | None, str]:
    """Process 1 record. Returns (output_dict_or_None, status).

    Status string một trong: "kept", "filtered_too_short", "filtered_no_domain".
    """
    raw_text = record.get("text", "")
    cleaned = clean(full_normalize(raw_text))

    if is_too_short(cleaned, min_chars=min_chars):
        return None, "filtered_too_short"

    law_id = record.get("law_id", "")
    metadata = parse_law_id(law_id)

    # Enrich law_title nếu corpus gốc trống.
    law_title = record.get("law_title") or metadata.synthetic_title()

    if apply_filter:
        match = classify_domain(law_id=law_id, text=cleaned)
        if match.domain is None:
            return None, "filtered_no_domain"
        domain = match.domain
        domain_via = "law_id" if match.via_law_id else "keyword"
    else:
        domain = "unknown"
        domain_via = "skipped"

    output = {
        "law_id": law_id,
        "law_title": law_title,
        "article_id": str(record.get("article_id", "")),
        "text": cleaned,
        "domain": domain,
        "metadata": {
            "type_code": metadata.type_code,
            "type_full": metadata.type_full,
            "issuer_code": metadata.issuer_code,
            "issuer_full": metadata.issuer_full,
            "year": metadata.year,
            "number": metadata.number,
            "domain_via": domain_via,
            "char_len": len(cleaned),
        },
    }

    # Preserve extra từ source nếu có.
    extra = record.get("extra") or {}
    if extra:
        output["metadata"]["source_extra"] = extra

    return output, "kept"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        logger.error("Hint: run `python scripts/01_pull_dataset.py --source sample` trước.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Filter 3-ngành: %s", "no (--no-filter)" if args.no_filter else "yes")
    logger.info("Limit: %s", args.limit or "all")

    counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    via_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as f_out:
        for record in iter_input_records(input_path, args.limit):
            counter["input"] += 1
            output, status = process_record(
                record,
                apply_filter=not args.no_filter,
                min_chars=args.min_chars,
            )
            counter[status] += 1
            if output is not None:
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                domain_counter[output["domain"]] += 1
                via_counter[output["metadata"]["domain_via"]] += 1

    # Stats
    logger.info("=" * 60)
    logger.info("Input records:           %d", counter["input"])
    logger.info("Filtered (too short):    %d", counter["filtered_too_short"])
    logger.info("Filtered (no domain):    %d", counter["filtered_no_domain"])
    logger.info("Kept:                    %d", counter["kept"])
    logger.info("-" * 60)
    if not args.no_filter and counter["kept"] > 0:
        logger.info("Domain breakdown:")
        for d, n in domain_counter.most_common():
            pct = 100 * n / counter["kept"]
            logger.info("  %-12s %6d  (%5.1f%%)", d, n, pct)
        logger.info("Detection method:")
        for v, n in via_counter.most_common():
            logger.info("  %-12s %6d", v, n)
    logger.info("=" * 60)

    if counter["kept"] == 0:
        logger.error("No articles kept — abort.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end eval script.

Workflow:
1. Load golden Q&A từ `data/raw/zalo_legal/qa.jsonl` (Phase 2 output)
   hoặc `data/eval/synthetic_qa.jsonl` (Phase 11 generate).
2. Run retrieval eval: Recall@k, MRR, nDCG (no LLM needed).
3. (Optional) Run generation eval: faithfulness + relevancy (LLM judge, slow).
4. Save results vào `data/eval/results/{timestamp}.json` + Markdown report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.api.dependencies import (  # noqa: E402
    get_hybrid_retriever,
    get_llm_client,
)
from src.config import PROJECT_ROOT  # noqa: E402
from src.eval.generation_eval import (  # noqa: E402
    GenerationSample,
    evaluate_generation,
)
from src.eval.retrieval_metrics import GoldenItem, evaluate_retrieval  # noqa: E402
from src.generation.rag_pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG eval")
    p.add_argument(
        "--golden",
        default=str(PROJECT_ROOT / "data" / "raw" / "zalo_legal" / "qa.jsonl"),
        help="Path to golden Q&A JSONL",
    )
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "eval" / "results"),
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument(
        "--gen-eval",
        action="store_true",
        help="Run generation eval (slow, dùng LLM judge).",
    )
    p.add_argument(
        "--gen-samples",
        type=int,
        default=20,
        help="Số samples để eval generation (nhỏ hơn cho tiết kiệm cost).",
    )
    return p.parse_args(argv)


def load_golden(path: Path, limit: int | None) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            items.append(GoldenItem.from_dict(json.loads(line)))
    return items


async def run_retrieval_eval(golden, top_k: int, enable_rerank: bool) -> dict:
    retriever = get_hybrid_retriever()

    def search(query: str):
        return retriever.search(query, top_k_final=top_k, enable_rerank=enable_rerank)

    logger.info("Running retrieval eval on %d queries (top_k=%d, rerank=%s)",
                len(golden), top_k, enable_rerank)
    t0 = time.time()
    result = evaluate_retrieval(golden, search, k_values=[1, 3, 5, 10])
    elapsed = time.time() - t0
    logger.info("Retrieval eval done in %.1fs", elapsed)
    return result.to_dict()


async def run_generation_eval(golden, n_samples: int, top_k: int) -> dict:
    """Generate answers cho N samples + judge."""
    retriever = get_hybrid_retriever()
    llm = get_llm_client()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=top_k)

    sample_set = golden[:n_samples]
    logger.info("Generating answers for %d samples...", len(sample_set))
    samples: list[GenerationSample] = []
    for i, item in enumerate(sample_set):
        result = await pipeline.ask(item.question, top_k=top_k)
        samples.append(
            GenerationSample(
                question=item.question,
                answer=result.answer,
                contexts=[h.text for h in result.sources[:5]],
            )
        )
        if (i + 1) % 5 == 0:
            logger.info("  Generated %d/%d", i + 1, len(sample_set))

    logger.info("Judging with %s ...", llm.model)
    judge_result = await evaluate_generation(samples, judge=llm)
    return judge_result.to_dict()


async def main_async(args) -> int:
    golden_path = Path(args.golden)
    if not golden_path.exists():
        logger.error("Golden file not found: %s", golden_path)
        return 1

    golden = load_golden(golden_path, args.limit)
    if not golden:
        logger.error("Empty golden set")
        return 1

    logger.info("Loaded %d golden items", len(golden))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"eval_{timestamp}.json"

    full_result: dict = {
        "timestamp": timestamp,
        "config": {
            "golden_path": str(golden_path),
            "top_k": args.top_k,
            "enable_rerank": not args.no_rerank,
            "n_golden": len(golden),
        },
    }

    # Retrieval eval (always)
    retrieval = await run_retrieval_eval(golden, args.top_k, not args.no_rerank)
    full_result["retrieval"] = retrieval

    print()
    print("=== RETRIEVAL ===")
    for k in [1, 3, 5, 10]:
        print(f"  Recall@{k}:    {retrieval['recall_at'].get(k, 0):.3f}")
        print(f"  Precision@{k}: {retrieval['precision_at'].get(k, 0):.3f}")
        print(f"  nDCG@{k}:      {retrieval['ndcg_at'].get(k, 0):.3f}")
    print(f"  MRR:          {retrieval['mrr']:.3f}")

    # Generation eval (optional, slow)
    if args.gen_eval:
        gen = await run_generation_eval(golden, args.gen_samples, args.top_k)
        full_result["generation"] = gen
        print()
        print("=== GENERATION ===")
        print(f"  N samples:        {gen['n_samples']}")
        print(f"  Avg faithfulness: {gen['avg_faithfulness']:.3f}")
        print(f"  Avg relevancy:    {gen['avg_relevancy']:.3f}")

    # Save
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", result_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

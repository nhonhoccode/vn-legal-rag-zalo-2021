"""Synthetic Q&A generator — Phase 11.

Strategy (per DATA_STRATEGY.md §6):
- Sample N chunks từ processed corpus.
- Cho mỗi chunk, LLM sinh 1-2 câu hỏi tự nhiên.
- Output JSONL format giống GoldenQA + flag `source: "synthetic"`.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass

from src.generation.llm_client import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


SYNTHETIC_QA_PROMPT = """Cho 1 điều luật, hãy sinh 2 câu hỏi tự nhiên mà người dùng có thể hỏi để truy xuất điều luật này.

Quy tắc:
1. Câu hỏi viết theo ngôn ngữ tự nhiên của người dân (KHÔNG copy từ ngữ trong điều luật).
2. Câu hỏi phải đủ cụ thể để chỉ trỏ tới điều luật này (không quá generic).
3. Mỗi câu hỏi 1 dòng, đánh số "1." và "2.".
4. KHÔNG giải thích, KHÔNG markdown, chỉ 2 dòng câu hỏi.

Ví dụ output:
1. Người lao động bị sa thải trái pháp luật được bồi thường gì?
2. Mức bồi thường tối thiểu khi đơn phương chấm dứt hợp đồng lao động là bao nhiêu?
"""


@dataclass(slots=True)
class SyntheticQAItem:
    """1 cặp Q&A synthetic."""

    question_id: str
    question: str
    law_id: str
    article_id: str
    source_chunk_text: str  # snippet để debug
    phrasing: str = "natural"  # natural | formal


def _parse_questions(text: str) -> list[str]:
    """Extract numbered questions từ LLM output."""
    lines = [line.strip() for line in text.split("\n")]
    questions: list[str] = []
    for line in lines:
        # Match "1. ...", "2. ...", "1) ..."
        if line and line[0].isdigit():
            # Remove number prefix
            cleaned = line.lstrip("0123456789.)").strip()
            if cleaned and len(cleaned) >= 5:
                questions.append(cleaned)
    return questions[:2]


async def generate_qa_for_chunk(
    chunk: dict,
    llm: LLMClient,
    *,
    base_index: int = 0,
) -> list[SyntheticQAItem]:
    """Generate ~2 Q&A items cho 1 chunk."""
    text = (chunk.get("text") or "").strip()
    if not text or len(text) < 50:
        return []

    law_id = chunk.get("law_id", "")
    article_id = chunk.get("article_id", "")

    messages = [
        LLMMessage(role="system", content=SYNTHETIC_QA_PROMPT),
        LLMMessage(role="user", content=text[:2000]),
    ]
    try:
        response = await llm.complete(messages, max_tokens=300, temperature=0.7)
    except Exception as e:  # noqa: BLE001
        logger.warning("Synthetic QA gen failed for %s: %s", law_id, e)
        return []

    questions = _parse_questions(response.text)
    items: list[SyntheticQAItem] = []
    for i, q in enumerate(questions):
        items.append(
            SyntheticQAItem(
                question_id=f"syn_{base_index}_{i}",
                question=q,
                law_id=law_id,
                article_id=article_id,
                source_chunk_text=text[:200],
                phrasing="natural" if i == 0 else "formal",
            )
        )
    return items


async def generate_synthetic_qa_set(
    chunks: Iterable[dict],
    llm: LLMClient,
    *,
    n_samples: int = 100,
    seed: int = 42,
) -> list[SyntheticQAItem]:
    """Sample N chunks → generate Q&A. Return list of SyntheticQAItem."""
    chunks_list = list(chunks)
    if not chunks_list:
        return []

    rng = random.Random(seed)
    sampled = rng.sample(chunks_list, min(n_samples, len(chunks_list)))

    all_items: list[SyntheticQAItem] = []
    for i, chunk in enumerate(sampled):
        items = await generate_qa_for_chunk(chunk, llm, base_index=i)
        all_items.extend(items)
        if (i + 1) % 10 == 0:
            logger.info("Generated %d/%d chunks → %d Q&A", i + 1, len(sampled), len(all_items))

    return all_items


def save_synthetic_qa_to_jsonl(items: Iterable[SyntheticQAItem], output_path) -> int:
    """Save format giống golden Q&A (compatible với eval pipeline)."""
    import pathlib

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            record = {
                "question_id": item.question_id,
                "question": item.question,
                "relevant_articles": [
                    {"law_id": item.law_id, "article_id": item.article_id}
                ],
                "source": "synthetic",
                "phrasing": item.phrasing,
                "source_chunk_text": item.source_chunk_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count

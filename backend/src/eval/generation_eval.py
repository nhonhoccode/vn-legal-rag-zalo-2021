"""Custom LLM-as-judge cho generation eval.

KHÔNG dùng ragas full (vì ragas mặc định OpenAI; setup workaround tốn).
Tự viết 2 metrics đơn giản:
- Faithfulness: answer có support bởi context không?
- Answer relevancy: answer có đúng câu hỏi không?

Mỗi metric = 1 LLM call → return score 0-1 + brief reason.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.generation.llm_client import LLMClient, LLMMessage

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """Bạn đánh giá xem câu trả lời có HOÀN TOÀN được hỗ trợ bởi context không.

Trả về duy nhất JSON format:
{"score": 0.0-1.0, "unsupported_claims": [str], "reason": str}

- score = 1.0: tất cả thông tin trong answer đều có trong context.
- score = 0.5: một phần answer được support, một phần bịa.
- score = 0.0: answer hoàn toàn bịa hoặc không liên quan context.
- unsupported_claims: list các câu trong answer không có trong context.

CHỈ trả JSON, không thêm text khác.
"""

RELEVANCY_PROMPT = """Bạn đánh giá xem câu trả lời có đúng câu hỏi không.

Trả về duy nhất JSON format:
{"score": 0.0-1.0, "reason": str}

- score = 1.0: answer trả lời đúng + đầy đủ câu hỏi.
- score = 0.5: answer relate nhưng thiếu/lệch.
- score = 0.0: answer không liên quan câu hỏi.

CHỈ trả JSON, không thêm text khác.
"""

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")


def _parse_json_response(text: str) -> dict:
    """Extract JSON từ LLM response (đôi khi có ```json wrapper)."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Find first {...} block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@dataclass(slots=True)
class GenerationJudgeResult:
    """Score per metric per sample."""

    question: str
    answer: str
    faithfulness: float = 0.0
    relevancy: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationEvalResult:
    n_samples: int
    avg_faithfulness: float
    avg_relevancy: float
    per_sample: list[GenerationJudgeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevancy": self.avg_relevancy,
            "per_sample": [
                {
                    "question": s.question,
                    "answer": s.answer[:200],
                    "faithfulness": s.faithfulness,
                    "relevancy": s.relevancy,
                    "unsupported_claims_count": len(s.unsupported_claims),
                }
                for s in self.per_sample
            ],
        }


@dataclass(slots=True)
class GenerationSample:
    """1 sample để eval: question + answer + retrieved contexts."""

    question: str
    answer: str
    contexts: list[str]


async def judge_faithfulness(
    sample: GenerationSample,
    judge: LLMClient,
) -> tuple[float, list[str], str]:
    context_block = "\n\n---\n\n".join(sample.contexts) if sample.contexts else "(empty)"
    user_msg = (
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{sample.question}\n\n"
        f"=== ANSWER ===\n{sample.answer}"
    )
    messages = [
        LLMMessage(role="system", content=FAITHFULNESS_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]
    try:
        response = await judge.complete(messages, max_tokens=512, temperature=0.0)
        data = _parse_json_response(response.text)
        score = float(data.get("score", 0.0))
        unsupported = data.get("unsupported_claims") or []
        reason = data.get("reason", "")
        return min(max(score, 0.0), 1.0), unsupported, reason
    except Exception as e:  # noqa: BLE001
        logger.warning("Faithfulness judge failed: %s", e)
        return 0.0, [], str(e)


async def judge_relevancy(
    sample: GenerationSample,
    judge: LLMClient,
) -> tuple[float, str]:
    user_msg = f"=== QUESTION ===\n{sample.question}\n\n=== ANSWER ===\n{sample.answer}"
    messages = [
        LLMMessage(role="system", content=RELEVANCY_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]
    try:
        response = await judge.complete(messages, max_tokens=256, temperature=0.0)
        data = _parse_json_response(response.text)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return min(max(score, 0.0), 1.0), reason
    except Exception as e:  # noqa: BLE001
        logger.warning("Relevancy judge failed: %s", e)
        return 0.0, str(e)


async def evaluate_generation(
    samples: Iterable[GenerationSample],
    judge: LLMClient,
) -> GenerationEvalResult:
    """Run faithfulness + relevancy judges trên list samples."""
    results: list[GenerationJudgeResult] = []
    for s in samples:
        faith, unsupported, faith_reason = await judge_faithfulness(s, judge)
        rel, rel_reason = await judge_relevancy(s, judge)
        results.append(
            GenerationJudgeResult(
                question=s.question,
                answer=s.answer,
                faithfulness=faith,
                relevancy=rel,
                unsupported_claims=unsupported,
                reasons={"faithfulness": faith_reason, "relevancy": rel_reason},
            )
        )

    n = len(results)
    if n == 0:
        return GenerationEvalResult(0, 0.0, 0.0)

    return GenerationEvalResult(
        n_samples=n,
        avg_faithfulness=sum(r.faithfulness for r in results) / n,
        avg_relevancy=sum(r.relevancy for r in results) / n,
        per_sample=results,
    )

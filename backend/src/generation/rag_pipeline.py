"""RAG pipeline: orchestrate retrieve → prompt → LLM → parse citations.

Single-turn ở Phase 8. Multi-turn + session sẽ ở Phase 9.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.retrieval.hybrid import HybridRetriever
from src.vectorstore.base import SearchHit

from .conversation import SessionManager
from .intent_classifier import (
    CHITCHAT_SYSTEM_PROMPT,
    TOO_LONG_MESSAGE,
    TOO_SHORT_MESSAGE,
    Intent,
    classify_intent,
)
from .llm_client import LLMClient, LLMMessage
from .prompts import (
    ParsedCitation,
    build_messages,
    build_messages_multiturn,
    build_rewrite_messages,
    is_refusal,
    parse_citations,
    query_likely_needs_rewrite,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AskResult:
    """Full output của 1 ask cycle."""

    query: str
    answer: str
    citations: list[ParsedCitation]
    sources: list[SearchHit]
    refused: bool
    latency_ms: int
    model: str
    retrieval_ms: int = 0
    generation_ms: int = 0
    rewrite_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    session_id: str | None = None
    standalone_query: str | None = None
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """End-to-end ask pipeline. Single-turn + multi-turn (qua SessionManager)."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        *,
        top_k: int = 5,
        min_top_score: float = 0.3,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        # Score threshold: nếu top-1 score thấp hơn → cảnh báo low confidence (không
        # hard-refuse vì system prompt đã handle).
        self.min_top_score = min_top_score
        self.session_manager = session_manager

    async def _rewrite_if_needed(
        self,
        query: str,
        history: list,  # list[Turn]
    ) -> tuple[str, int]:
        """Standalone query rewriting. Return (rewritten, rewrite_ms).

        Skip nếu history rỗng hoặc query không chứa reference words (heuristic).
        """
        if not history or not query_likely_needs_rewrite(query):
            return query, 0

        try:
            t0 = time.time()
            messages = build_rewrite_messages(query, history)
            response = await self.llm.complete(
                messages, max_tokens=256, temperature=0.0
            )
            rewrite_ms = int((time.time() - t0) * 1000)
            rewritten = response.text.strip().strip('"').strip("'")
            if not rewritten or len(rewritten) > 500:
                # Sanity guard — fallback to original
                logger.warning("Rewrite returned bad output, fallback to original")
                return query, rewrite_ms
            return rewritten, rewrite_ms
        except Exception as e:  # noqa: BLE001
            logger.warning("Rewrite failed: %s — fallback to original query", e)
            return query, 0

    async def _handle_chitchat(
        self,
        query: str,
        *,
        max_tokens: int,
        temperature: float,
        t0: float,
        session_id: str | None,
    ) -> AskResult:
        """Chit-chat path: LLM only, no retrieval. Tone tự nhiên."""
        tg0 = time.time()
        response = await self.llm.complete(
            [
                LLMMessage(role="system", content=CHITCHAT_SYSTEM_PROMPT),
                LLMMessage(role="user", content=query),
            ],
            max_tokens=max_tokens,
            temperature=max(temperature, 0.3),  # ấm hơn cho chit-chat
        )
        generation_ms = int((time.time() - tg0) * 1000)
        total_ms = int((time.time() - t0) * 1000)
        return AskResult(
            query=query,
            answer=response.text,
            citations=[],
            sources=[],
            refused=False,
            latency_ms=total_ms,
            model=response.model,
            retrieval_ms=0,
            generation_ms=generation_ms,
            rewrite_ms=0,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            session_id=session_id,
            metadata={"intent": Intent.CHITCHAT.value},
        )

    def _short_circuit_result(
        self, query: str, answer: str, intent: Intent, t0: float, session_id: str | None
    ) -> AskResult:
        """Trả ngay không gọi LLM — cho input quá ngắn/quá dài."""
        return AskResult(
            query=query,
            answer=answer,
            citations=[],
            sources=[],
            refused=False,
            latency_ms=int((time.time() - t0) * 1000),
            model="rule-based",
            session_id=session_id,
            metadata={"intent": intent.value},
        )

    async def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        domain: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        session_id: str | None = None,
    ) -> AskResult:
        """Non-streaming ask. Nếu session_id → multi-turn mode."""
        t0 = time.time()

        # 0. Intent classification — skip RAG cho chit-chat / invalid input
        intent = classify_intent(query)
        if intent == Intent.TOO_SHORT:
            return self._short_circuit_result(query, TOO_SHORT_MESSAGE, intent, t0, session_id)
        if intent == Intent.TOO_LONG:
            return self._short_circuit_result(query, TOO_LONG_MESSAGE, intent, t0, session_id)
        if intent == Intent.CHITCHAT:
            return await self._handle_chitchat(
                query,
                max_tokens=max_tokens,
                temperature=temperature,
                t0=t0,
                session_id=session_id,
            )

        # 1. Load history (if multi-turn)
        history: list = []
        if session_id and self.session_manager:
            history = await self.session_manager.get_history(session_id)

        # 2. Standalone query rewrite (skip nếu single-turn hoặc query đã độc lập)
        standalone_query, rewrite_ms = await self._rewrite_if_needed(query, history)

        # 3. Retrieval (dùng standalone_query)
        tr0 = time.time()
        hits = self.retriever.search(
            standalone_query,
            top_k_final=top_k or self.top_k,
            where={"domain": domain} if domain else None,
        )
        retrieval_ms = int((time.time() - tr0) * 1000)

        # 4. LLM call — pass history nếu có
        if history:
            messages = build_messages_multiturn(query, hits, history=history)
        else:
            messages = build_messages(query, hits)

        tg0 = time.time()
        response = await self.llm.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generation_ms = int((time.time() - tg0) * 1000)

        # 5. Parse citations
        citations = parse_citations(response.text, hits)
        refused = is_refusal(response.text) or not hits

        # 6. Persist history nếu multi-turn
        if session_id and self.session_manager:
            await self.session_manager.append_pair(
                session_id,
                user_query=query,
                assistant_answer=response.text,
                citations=[
                    {
                        "law_id": c.law_id,
                        "article_id": c.article_id,
                        "khoan_id": c.khoan_id,
                    }
                    for c in citations
                ],
            )

        total_ms = int((time.time() - t0) * 1000)

        return AskResult(
            query=query,
            answer=response.text,
            citations=citations,
            sources=hits,
            refused=refused,
            latency_ms=total_ms,
            model=response.model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            rewrite_ms=rewrite_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            session_id=session_id,
            standalone_query=standalone_query if standalone_query != query else None,
            metadata={
                "top_score": hits[0].score if hits else 0.0,
                "low_confidence": (hits[0].score < self.min_top_score) if hits else True,
                "n_hits": len(hits),
                "n_citations": len(citations),
                "n_citations_matched": sum(1 for c in citations if c.matched),
                "n_history_turns": len(history),
                "rewritten": standalone_query != query,
            },
        )

    async def ask_stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        domain: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        session_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Streaming ask. Yields events:

        - `{"type": "progress", "stage": "retrieval|reranking|generation"}`
        - `{"type": "sources", "sources": [...]}` (1 lần)
        - `{"type": "token", "value": "..."}`    (nhiều lần)
        - `{"type": "done", "citations": [...], "metadata": {...}}` (1 lần, cuối)
        """
        t0 = time.time()

        # 0. Intent routing — short-circuit cho chit-chat/invalid input
        intent = classify_intent(query)
        if intent == Intent.TOO_SHORT:
            yield {"type": "token", "value": TOO_SHORT_MESSAGE}
            yield {
                "type": "done",
                "citations": [], "refused": False,
                "latency_ms": int((time.time() - t0) * 1000),
                "model": "rule-based", "n_hits": 0, "session_id": session_id,
                "intent": intent.value,
            }
            return
        if intent == Intent.TOO_LONG:
            yield {"type": "token", "value": TOO_LONG_MESSAGE}
            yield {
                "type": "done",
                "citations": [], "refused": False,
                "latency_ms": int((time.time() - t0) * 1000),
                "model": "rule-based", "n_hits": 0, "session_id": session_id,
                "intent": intent.value,
            }
            return
        if intent == Intent.CHITCHAT:
            yield {"type": "progress", "stage": "generation"}
            collected: list[str] = []
            async for chunk in self.llm.complete_stream(
                [
                    LLMMessage(role="system", content=CHITCHAT_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=query),
                ],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.3),
            ):
                collected.append(chunk)
                yield {"type": "token", "value": chunk}
            yield {
                "type": "done",
                "citations": [], "refused": False,
                "latency_ms": int((time.time() - t0) * 1000),
                "model": self.llm.model, "n_hits": 0, "session_id": session_id,
                "intent": intent.value,
            }
            return

        # 1. Load history + rewrite (multi-turn)
        history: list = []
        if session_id and self.session_manager:
            history = await self.session_manager.get_history(session_id)
        standalone_query, _ = await self._rewrite_if_needed(query, history)

        # 2. Retrieval
        yield {"type": "progress", "stage": "retrieval"}
        hits = self.retriever.search(
            standalone_query,
            top_k_final=top_k or self.top_k,
            where={"domain": domain} if domain else None,
        )

        yield {
            "type": "sources",
            "sources": [
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "law_id": h.metadata.get("law_id", ""),
                    "law_title": h.metadata.get("law_title", ""),
                    "article_id": h.metadata.get("article_id", ""),
                    "text": h.text[:200] + "..." if len(h.text) > 200 else h.text,
                }
                for h in hits
            ],
        }

        # 3. LLM streaming
        yield {"type": "progress", "stage": "generation"}
        messages = (
            build_messages_multiturn(query, hits, history=history)
            if history
            else build_messages(query, hits)
        )
        collected: list[str] = []
        async for chunk in self.llm.complete_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            collected.append(chunk)
            yield {"type": "token", "value": chunk}

        full_answer = "".join(collected)
        citations = parse_citations(full_answer, hits)
        refused = is_refusal(full_answer) or not hits

        # 4. Persist history nếu multi-turn
        if session_id and self.session_manager:
            await self.session_manager.append_pair(
                session_id,
                user_query=query,
                assistant_answer=full_answer,
                citations=[
                    {"law_id": c.law_id, "article_id": c.article_id, "khoan_id": c.khoan_id}
                    for c in citations
                ],
            )

        total_ms = int((time.time() - t0) * 1000)

        yield {
            "type": "done",
            "citations": [
                {
                    "text_index": c.text_index,
                    "law_id": c.law_id,
                    "law_title": c.law_title,
                    "article_id": c.article_id,
                    "khoan_id": c.khoan_id,
                    "matched": c.matched,
                }
                for c in citations
            ],
            "refused": refused,
            "latency_ms": total_ms,
            "model": self.llm.model,
            "n_hits": len(hits),
            "session_id": session_id,
        }

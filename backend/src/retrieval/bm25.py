"""BM25 retrieval qua `rank_bm25` + Vietnamese tokenizer (`pyvi`).

BM25 quan trọng cho luật VN vì identifier ("Điều 41 BLLĐ", "45/2019/QH14")
mà dense embedding không bắt được tốt.

Index lưu pickle ở `data/processed/bm25.pkl`.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.vectorstore.base import SearchHit


def tokenize_vi(text: str) -> list[str]:
    """Vietnamese tokenize qua pyvi.

    pyvi.ViTokenizer.tokenize joins compound words với underscore:
        "kinh tế thị trường" → "kinh_tế thị_trường"
    Sau đó split() ra tokens.

    Cũng lower + strip punctuation đơn giản.
    """
    from pyvi import ViTokenizer

    text = text.lower()
    # Thay punct bằng space (giữ chữ cái VN + số + _ vì pyvi dùng underscore)
    text = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]", " ", text)
    tokenized = ViTokenizer.tokenize(text)
    return [t for t in tokenized.split() if t.strip()]


@dataclass(slots=True)
class BM25Document:
    chunk_id: str
    text: str
    metadata: dict


class BM25Retriever:
    """In-memory BM25 retriever. Build từ list of chunks; persist optional."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[BM25Document] = []

    def index(self, chunks: list[dict]) -> None:
        """Index từ list of chunk dicts (cần `chunk_id`, `text`, `metadata`)."""
        self._docs = [
            BM25Document(
                chunk_id=c["chunk_id"],
                text=c.get("text", ""),
                metadata={
                    "law_id": c.get("law_id", ""),
                    "law_title": c.get("law_title", ""),
                    "article_id": c.get("article_id", ""),
                    "domain": c.get("domain", ""),
                    **(c.get("metadata") or {}),
                },
            )
            for c in chunks
        ]
        tokenized_corpus = [tokenize_vi(d.text) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 30) -> list[SearchHit]:
        if self._bm25 is None or not self._docs:
            return []
        tokens = tokenize_vi(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Top-k indices.
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            SearchHit(
                chunk_id=self._docs[i].chunk_id,
                text=self._docs[i].text,
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"bm25": self._bm25, "docs": self._docs}, f)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)  # noqa: S301 — trusted local file
        self._bm25 = data["bm25"]
        self._docs = data["docs"]

    def count(self) -> int:
        return len(self._docs)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    top_k: int = 10,
    k: int = 60,
) -> list[SearchHit]:
    """Reciprocal Rank Fusion để merge multiple ranked lists.

    Score per doc = sum(1 / (k + rank)) qua các rankings.

    Reference: Cormack et al., 2009.
    """
    fused_score: dict[str, float] = {}
    representative: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = hit.chunk_id
            fused_score[cid] = fused_score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in representative:
                representative[cid] = hit

    sorted_ids = sorted(fused_score, key=lambda c: -fused_score[c])[:top_k]
    return [
        SearchHit(
            chunk_id=cid,
            text=representative[cid].text,
            score=fused_score[cid],
            metadata=representative[cid].metadata,
        )
        for cid in sorted_ids
    ]

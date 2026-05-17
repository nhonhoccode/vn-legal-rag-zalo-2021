"""Zalo AI Legal Text Retrieval 2021 — multi-source loader.

Hai cách lấy data:

1. **HuggingFace** (`ZaloLegalHFSource`): try a list of known HF dataset names
   community-hosted (Zalo không upload official lên HF — phải dùng mirror).
2. **Local files** (`ZaloLegalLocalSource`): user download manual từ Zalo / GitHub
   rồi point đường dẫn corpus.json + qa.json.

Plus utility helpers `save_articles_to_jsonl` / `save_qa_to_jsonl`.

Decision (DECISIONS.md D2): Zalo Legal 2021 = primary source. Filter theo
3 ngành (Lao động + Doanh nghiệp/Đầu tư + Hình sự) sẽ làm ở **Phase 3**
(preprocessing) — Phase 2 chỉ pull raw.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import Article, GoldenQA, IngestionSource

logger = logging.getLogger(__name__)

# Known HF dataset pairs (corpus_name, qa_name) — community mirrors.
# Verified real datasets (2026-05-06).
# User có thể override qua `--corpus-dataset` / `--qa-dataset` CLI args.
KNOWN_HF_DATASET_PAIRS: tuple[tuple[str, str | None], ...] = (
    # Primary: NghiemAbe — 61k articles + 3k Q&A, flat schema, verified
    ("NghiemAbe/Legal-Corpus-Zalo", "NghiemAbe/Legal_Train_Zalo"),
)

# Backward compat: list of corpus names only (for older API)
KNOWN_HF_DATASETS: tuple[str, ...] = tuple(c for c, _ in KNOWN_HF_DATASET_PAIRS)


# ============================================================
# HuggingFace source
# ============================================================
class ZaloLegalHFSource(IngestionSource):
    """Load Zalo Legal 2021 từ HuggingFace datasets.

    Zalo Legal 2021 trên HF được host **dạng cặp** (corpus dataset riêng + QA
    dataset riêng), nên class này nhận 2 dataset names. Nếu omit, auto try
    `KNOWN_HF_DATASET_PAIRS`.

    Verified pair (2026-05-06):
        corpus: `NghiemAbe/Legal-Corpus-Zalo` — 61,425 articles, schema flat
                {law_id, article_id, title, text}
        qa:     `NghiemAbe/Legal_Train_Zalo` — 3,298 Q&A, schema flat
                {question_id, question, law_id, article_id, content}
    """

    # Backward-compat alias: nhận 1 string `dataset_name` → coi như corpus
    def __init__(
        self,
        corpus_dataset: str | None = None,
        qa_dataset: str | None = None,
        *,
        dataset_name: str | None = None,  # deprecated alias
    ) -> None:
        self.corpus_dataset = corpus_dataset or dataset_name
        self.qa_dataset = qa_dataset
        self._corpus_ds = None
        self._qa_ds = None

    @property
    def name(self) -> str:
        return f"zalo_legal_hf(corpus={self.corpus_dataset or 'auto'}, qa={self.qa_dataset or 'auto'})"

    def _load_corpus_ds(self):  # noqa: ANN202
        if self._corpus_ds is not None:
            return self._corpus_ds

        from datasets import load_dataset

        candidates: tuple[str, ...]
        if self.corpus_dataset:
            candidates = (self.corpus_dataset,)
        else:
            candidates = tuple(c for c, _ in KNOWN_HF_DATASET_PAIRS)

        last_err: Exception | None = None
        for name in candidates:
            try:
                logger.info("Loading HF corpus dataset: %s", name)
                ds = load_dataset(name)
                self._corpus_ds = ds
                self.corpus_dataset = name
                # Auto-pair QA if not specified
                if not self.qa_dataset:
                    for c, q in KNOWN_HF_DATASET_PAIRS:
                        if c == name and q:
                            self.qa_dataset = q
                            break
                return ds
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load corpus %s: %s", name, e)
                last_err = e

        raise RuntimeError(
            f"Could not load any HF corpus dataset from {candidates}. Last error: {last_err}"
        )

    def _load_qa_ds(self):  # noqa: ANN202
        if self._qa_ds is not None:
            return self._qa_ds
        if not self.qa_dataset:
            # Trigger corpus load to auto-set qa_dataset
            self._load_corpus_ds()
        if not self.qa_dataset:
            return None

        from datasets import load_dataset

        try:
            logger.info("Loading HF QA dataset: %s", self.qa_dataset)
            ds = load_dataset(self.qa_dataset)
            self._qa_ds = ds
            return ds
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load QA dataset %s: %s", self.qa_dataset, e)
            return None

    @staticmethod
    def _pick_split(ds, preferred: tuple[str, ...]):  # noqa: ANN001, ANN205
        """Chọn split phù hợp từ HF DatasetDict."""
        split_names = list(ds.keys()) if hasattr(ds, "keys") else []
        for s in preferred:
            if s in split_names:
                return ds[s]
        if split_names:
            logger.warning("No preferred split found in %s; using %s", preferred, split_names[0])
            return ds[split_names[0]]
        return ds  # plain Dataset

    @staticmethod
    def _iter_articles_from_dataset(ds) -> Iterator[Article]:  # noqa: ANN001
        """Adapter: convert HF corpus record → Article. Hỗ trợ 2 schema phổ biến."""
        split = ZaloLegalHFSource._pick_split(ds, ("corpus", "train", "law", "documents"))

        for record in split:
            # Schema A (NghiemAbe/Legal-Corpus-Zalo): flat {law_id, article_id, title, text}
            if "law_id" in record and "article_id" in record and "text" in record:
                title = record.get("title", "")
                body = record.get("text", "")
                # Combine title + body. Title thường dạng "Điều N. ..." → đặt đầu.
                full_text = f"{title}\n{body}".strip() if title else body
                yield Article(
                    law_id=str(record["law_id"]),
                    law_title=record.get("law_title", ""),
                    article_id=str(record["article_id"]),
                    text=full_text,
                    extra={"title": title} if title else {},
                )
            # Schema B (Zalo gốc): nested {law_id, articles: [{article_id, text}]}
            elif "articles" in record and isinstance(record["articles"], list):
                law_id = str(record.get("law_id") or record.get("id", ""))
                law_title = record.get("law_title") or record.get("title", "")
                for art in record["articles"]:
                    yield Article(
                        law_id=law_id,
                        law_title=law_title,
                        article_id=str(art.get("article_id") or art.get("id", "")),
                        text=art.get("text", ""),
                    )
            else:
                logger.warning("Skipping record với schema lạ: keys=%s", list(record.keys()))

    @staticmethod
    def _iter_qa_from_dataset(ds) -> Iterator[GoldenQA]:  # noqa: ANN001
        """Adapter: convert HF QA record → GoldenQA. Hỗ trợ 2 schema phổ biến."""
        split = ZaloLegalHFSource._pick_split(ds, ("train", "qa", "questions", "test", "validation"))

        for item in split:
            # Schema A (NghiemAbe/Legal_Train_Zalo): flat — relevant pointer = (law_id, article_id)
            if "question" in item and "law_id" in item and "article_id" in item:
                yield GoldenQA(
                    question_id=str(item.get("question_id") or item.get("id", "")),
                    question=item["question"],
                    relevant_articles=[
                        {
                            "law_id": str(item["law_id"]),
                            "article_id": str(item["article_id"]),
                        }
                    ],
                )
            # Schema B: explicit relevant_articles list
            elif "question" in item and "relevant_articles" in item:
                yield GoldenQA(
                    question_id=str(item.get("question_id") or item.get("id", "")),
                    question=item["question"],
                    relevant_articles=item["relevant_articles"] or [],
                )
            else:
                logger.warning("Skipping QA record với schema lạ: keys=%s", list(item.keys()))

    def fetch_corpus(self) -> Iterator[Article]:
        ds = self._load_corpus_ds()
        yield from self._iter_articles_from_dataset(ds)

    def fetch_qa(self) -> Iterator[GoldenQA]:
        ds = self._load_qa_ds()
        if ds is None:
            return
        yield from self._iter_qa_from_dataset(ds)


# ============================================================
# Local file source
# ============================================================
class ZaloLegalLocalSource(IngestionSource):
    """Load Zalo Legal 2021 từ file local đã download manual.

    Hỗ trợ:
    - corpus: `.jsonl` (1 record/line) hoặc `.json` (list of records)
    - qa: same

    Schema kỳ vọng giống Zalo gốc:
        corpus.json: [{"law_id", "articles": [{"article_id", "text"}]}, ...]
        qa.json:     [{"question_id", "question", "relevant_articles": [...]}, ...]
    """

    def __init__(self, corpus_path: Path | str, qa_path: Path | str | None = None) -> None:
        self.corpus_path = Path(corpus_path)
        self.qa_path = Path(qa_path) if qa_path else None

    @property
    def name(self) -> str:
        return f"zalo_legal_local({self.corpus_path.name})"

    def fetch_corpus(self) -> Iterator[Article]:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        for record in _read_records(self.corpus_path):
            yield from _record_to_articles(record)

    def fetch_qa(self) -> Iterator[GoldenQA]:
        if not self.qa_path:
            return
        if not self.qa_path.exists():
            logger.warning("QA file not found: %s — skipping", self.qa_path)
            return

        for item in _read_records(self.qa_path):
            yield GoldenQA(
                question_id=str(item.get("question_id") or item.get("id", "")),
                question=item.get("question", ""),
                relevant_articles=item.get("relevant_articles") or [],
            )


def _read_records(path: Path) -> Iterator[dict]:
    """Đọc file jsonl hoặc json (list)."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "items" in data:
            yield from data["items"]
        else:
            raise ValueError(f"Unsupported JSON shape in {path}")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix} (need .json/.jsonl)")


def _record_to_articles(record: dict) -> Iterator[Article]:
    """Convert raw record → Articles. Support 2 schemas (nested or flat)."""
    if "articles" in record and isinstance(record["articles"], list):
        law_id = str(record.get("law_id") or record.get("id", ""))
        law_title = record.get("law_title") or record.get("title", "")
        for art in record["articles"]:
            yield Article(
                law_id=law_id,
                law_title=law_title,
                article_id=str(art.get("article_id") or art.get("id", "")),
                text=art.get("text", ""),
            )
    elif "law_id" in record and "article_id" in record:
        yield Article(
            law_id=str(record["law_id"]),
            law_title=record.get("law_title", ""),
            article_id=str(record["article_id"]),
            text=record.get("text", ""),
        )
    else:
        logger.warning("Skipping record with unknown schema: keys=%s", list(record.keys()))


# ============================================================
# Save helpers
# ============================================================
def save_articles_to_jsonl(articles: Iterable[Article], output_path: Path | str) -> int:
    """Lưu articles vào JSONL. Return số records.

    Tiếng Việt được giữ raw UTF-8 (`ensure_ascii=False`) — KHÔNG escape \\uXXXX.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def save_qa_to_jsonl(qa_items: Iterable[GoldenQA], output_path: Path | str) -> int:
    """Lưu Q&A vào JSONL. Return số records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for qa in qa_items:
            f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count

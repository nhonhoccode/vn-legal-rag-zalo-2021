"""Abstract base classes + dataclasses cho data ingestion.

Mỗi `IngestionSource` cài đặt cách kéo `Article` corpus + `GoldenQA` Q&A
từ một nguồn cụ thể (HF, local file, sample mock, ...). Output schema cố định
để các phase sau (preprocessing, chunking) không phụ thuộc nguồn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Article:
    """Một điều luật trong corpus pháp luật VN.

    Đây là đơn vị raw — chunking thật sự (cắt theo Khoản nếu cần)
    sẽ ở Phase 4. Phase 2 giữ nguyên cấu trúc gốc của dataset.
    """

    law_id: str
    law_title: str
    article_id: str
    text: str
    extra: dict = field(default_factory=dict)  # giữ field bonus của dataset gốc

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class GoldenQA:
    """Một cặp Q&A có golden relevant articles (cho eval).

    `relevant_articles` là list of {law_id, article_id} — pointer tới Articles
    trong corpus.
    """

    question_id: str
    question: str
    relevant_articles: list[dict]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionSource(ABC):
    """Abstract source cho legal data."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên source (cho logging)."""

    @abstractmethod
    def fetch_corpus(self) -> Iterator[Article]:
        """Iterate qua tất cả articles."""

    @abstractmethod
    def fetch_qa(self) -> Iterator[GoldenQA]:
        """Iterate qua Q&A pairs. Có thể yield nothing nếu source không có Q&A."""

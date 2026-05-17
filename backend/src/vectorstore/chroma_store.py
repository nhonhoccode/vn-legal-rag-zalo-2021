"""ChromaDB persistent vector store wrapper.

Theo Decision 4: ChromaDB cho dev/MVP, Qdrant nếu sau cần scale > 1M vectors.
Cosine similarity (HNSW index, default Chroma).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .base import SearchHit, VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Wrapper quanh `chromadb.PersistentClient`."""

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "legal_vn",
        distance: str = "cosine",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.distance = distance
        self._client = None
        self._collection = None

    @property
    def name(self) -> str:
        return f"chroma({self.collection_name}@{self.persist_dir})"

    def _get_client(self):  # noqa: ANN202
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_or_create_collection(self):  # noqa: ANN202
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )
        return self._collection

    def count(self) -> int:
        return self._get_or_create_collection().count()

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        collection = self._get_or_create_collection()
        # Sanitize metadata: ChromaDB chỉ accept str/int/float/bool/None values.
        sanitized = [_sanitize_metadata(m) for m in metadatas]
        collection.add(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
            metadatas=sanitized,
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> None:
        """Như add() nhưng overwrite nếu id đã có."""
        if not ids:
            return
        collection = self._get_or_create_collection()
        sanitized = [_sanitize_metadata(m) for m in metadatas]
        collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
            metadatas=sanitized,
        )

    def query(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[SearchHit]:
        collection = self._get_or_create_collection()
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where,
        )
        # Chroma format: dict of arrays, indexed by query_idx (0 since we passed 1 query).
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        hits: list[SearchHit] = []
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            # Chroma cosine "distance" = 1 - cosine_similarity → score = 1 - distance.
            score = 1.0 - distance if self.distance == "cosine" else -distance
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=documents[i] if i < len(documents) else "",
                    score=float(score),
                    metadata=metadatas[i] if i < len(metadatas) else {},
                )
            )
        return hits

    def delete_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Delete collection failed (maybe non-existent): %s", e)
        self._collection = None


def _sanitize_metadata(metadata: dict) -> dict:
    """ChromaDB metadata values: str | int | float | bool (KHÔNG accept None).

    - Drop key có value None.
    - Convert nested dicts/lists thành str(...).
    - Empty dict → thêm placeholder field (ChromaDB reject {}).
    """
    out: dict = {}
    for k, v in metadata.items():
        if v is None:
            continue  # Chroma reject None
        if isinstance(v, str | int | float | bool):
            out[k] = v
        else:
            out[k] = str(v)
    if not out:
        out["_placeholder"] = ""
    return out

"""Index chunks → ChromaDB.

Hai use cases:

1. **Local embed** (small corpus, slow CPU): chunks → encode (small model) → ChromaDB.
2. **Kaggle import**: user đã chạy notebook Kaggle → output `chroma_db.zip`.
   Unzip vào `data/chroma_db/` rồi script này chỉ verify.

Default (small corpus): tự embed dùng `make_small_embedder` để smoke test.
For real bge-m3 → chạy notebook Kaggle.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config import PROJECT_ROOT, settings  # noqa: E402
from src.embeddings.sentence_transformer_embedder import (  # noqa: E402
    STEmbedder,
    make_default_embedder,
    make_small_embedder,
)
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index chunks → ChromaDB")
    p.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"),
    )
    p.add_argument(
        "--persist-dir",
        default=str(PROJECT_ROOT / "data" / "chroma_db"),
    )
    p.add_argument("--collection", default=settings.chroma_collection)
    p.add_argument(
        "--model",
        choices=["default", "small"],
        default="small",
        help="default = bge-m3 (2.27GB, slow trên CPU); small = MiniLM-L3 (17MB, smoke test).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete collection trước khi index.",
    )
    return p.parse_args(argv)


def load_chunks(path: Path, limit: int | None) -> list[dict]:
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def make_embedder(choice: str) -> STEmbedder:
    if choice == "default":
        return make_default_embedder(device=settings.embedding_device)
    return make_small_embedder(device=settings.embedding_device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input not found: %s — chạy `python scripts/03_chunk.py` trước.", input_path)
        return 1

    chunks = load_chunks(input_path, args.limit)
    if not chunks:
        logger.error("No chunks loaded.")
        return 1
    logger.info("Loaded %d chunks", len(chunks))

    store = ChromaStore(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
    )

    if args.reset:
        logger.info("Resetting collection %s", args.collection)
        store.delete_collection()

    embedder = make_embedder(args.model)
    logger.info("Using embedder: %s", embedder.name)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "law_id": c.get("law_id", ""),
            "law_title": (c.get("law_title", "") or "")[:200],
            "article_id": c.get("article_id", ""),
            "domain": c.get("domain", ""),
            **{
                k: v
                for k, v in (c.get("metadata") or {}).items()
                if v is None or isinstance(v, str | int | float | bool)
            },
        }
        for c in chunks
    ]

    # Encode (sequential — let sentence_transformers handle internal batching)
    embeddings = embedder.encode(texts, batch_size=args.batch_size, show_progress=True)
    logger.info("Encoded shape: %s", embeddings.shape)

    # Add (in chunks of 500 để tránh ChromaDB OOM with very large batches)
    add_batch = 500
    for i in range(0, len(ids), add_batch):
        store.add(
            ids=ids[i : i + add_batch],
            documents=texts[i : i + add_batch],
            embeddings=embeddings[i : i + add_batch].tolist(),
            metadatas=metadatas[i : i + add_batch],
        )

    final_count = store.count()
    logger.info("=" * 60)
    logger.info("Indexed: %d / %d", final_count, len(chunks))
    logger.info("Persist dir: %s", args.persist_dir)
    logger.info("=" * 60)

    if final_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

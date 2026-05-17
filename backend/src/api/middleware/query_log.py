"""Query logging → SQLite (Phase 13).

Lightweight monitoring:
- Log mỗi request /api/ask vào `data/app.db`.
- Track: timestamp, user_id, query, model, latency, n_hits, refused.
- Async background write (không block response).
- Gọi trực tiếp trong route handler (KHÔNG dùng BaseHTTPMiddleware — incompat streaming).

Schema (auto-create):
    queries(id, ts, user_id, endpoint, query, model, latency_ms, n_hits, status, refused, error)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    user_id TEXT,
    endpoint TEXT NOT NULL,
    query TEXT,
    model TEXT,
    latency_ms INTEGER,
    n_hits INTEGER,
    status INTEGER,
    refused INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id);
"""


class QueryLogger:
    """SQLite-backed query log. Thread-safe writes."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._lock = asyncio.Lock()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        try:
            yield conn
        finally:
            conn.close()

    async def log(self, entry: dict[str, Any]) -> None:
        """Async insert. Swallow errors (don't break request flow)."""
        try:
            async with self._lock:
                await asyncio.to_thread(self._insert_sync, entry)
        except Exception as e:  # noqa: BLE001
            logger.warning("Query log failed: %s", e)

    def _insert_sync(self, entry: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO queries
                (ts, user_id, endpoint, query, model, latency_ms, n_hits, status, refused, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("ts", int(time.time())),
                    entry.get("user_id"),
                    entry.get("endpoint", ""),
                    (entry.get("query") or "")[:1000],
                    entry.get("model"),
                    entry.get("latency_ms"),
                    entry.get("n_hits"),
                    entry.get("status"),
                    1 if entry.get("refused") else 0,
                    (entry.get("error") or "")[:500] if entry.get("error") else None,
                ),
            )

    def stats(self, since_ts: int | None = None) -> dict:
        """Quick aggregate stats."""
        with self._connect() as conn:
            cur = conn.cursor()
            where = f"WHERE ts >= {int(since_ts)}" if since_ts else ""
            cur.execute(f"SELECT COUNT(*), AVG(latency_ms), SUM(refused) FROM queries {where}")
            row = cur.fetchone()
            return {
                "total_queries": row[0] or 0,
                "avg_latency_ms": int(row[1] or 0),
                "refused_count": row[2] or 0,
            }


# Middleware version removed — incompat với streaming SSE.
# Route handlers (vd /api/ask) gọi `get_query_logger().log(...)` trực tiếp.

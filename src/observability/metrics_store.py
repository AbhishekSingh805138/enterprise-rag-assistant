"""SQLite-backed persistence for per-query cost and latency metrics.

Stores metrics in the same database as the LangGraph checkpointer but uses
a separate connection. The query_metrics table is created idempotently on
first access.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings

from src.observability.cost_callback import QueryMetrics

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS query_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    thread_id   TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    retriever   TEXT    NOT NULL,
    prompt_tok  INTEGER NOT NULL DEFAULT 0,
    compl_tok   INTEGER NOT NULL DEFAULT 0,
    total_tok   INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    latency_ms  REAL    NOT NULL DEFAULT 0.0
);
"""

COST_BUDGET = 0.02  # USD per query — PRD target


class MetricsStore:
    """Thin wrapper around a SQLite table for query metrics."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def record(self, m: QueryMetrics) -> None:
        """Insert a single query's metrics."""
        self._conn.execute(
            "INSERT INTO query_metrics "
            "(ts, thread_id, question, mode, retriever, prompt_tok, compl_tok, total_tok, cost_usd, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                m.thread_id,
                m.question_preview,
                m.mode,
                m.retriever_strategy,
                m.prompt_tokens,
                m.completion_tokens,
                m.total_tokens,
                m.estimated_cost_usd,
                m.latency_ms,
            ),
        )
        self._conn.commit()

    def query_recent(self, n: int = 20) -> list[dict]:
        """Return the last *n* query metrics, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM query_metrics ORDER BY id DESC LIMIT ?", (n,)
        )
        return [dict(row) for row in cur.fetchall()]

    def summary(self, n: int | None = None) -> dict:
        """Aggregate statistics over the last *n* queries (or all if None)."""
        if n is not None:
            sql = (
                "SELECT COUNT(*) AS cnt, "
                "COALESCE(SUM(cost_usd), 0) AS total_cost, "
                "COALESCE(AVG(cost_usd), 0) AS avg_cost, "
                "COALESCE(AVG(latency_ms), 0) AS avg_latency, "
                "COALESCE(SUM(total_tok), 0) AS total_tokens, "
                "COALESCE(SUM(CASE WHEN cost_usd > ? THEN 1 ELSE 0 END), 0) AS over_budget "
                "FROM (SELECT * FROM query_metrics ORDER BY id DESC LIMIT ?)"
            )
            row = self._conn.execute(sql, (COST_BUDGET, n)).fetchone()
        else:
            sql = (
                "SELECT COUNT(*) AS cnt, "
                "COALESCE(SUM(cost_usd), 0) AS total_cost, "
                "COALESCE(AVG(cost_usd), 0) AS avg_cost, "
                "COALESCE(AVG(latency_ms), 0) AS avg_latency, "
                "COALESCE(SUM(total_tok), 0) AS total_tokens, "
                "COALESCE(SUM(CASE WHEN cost_usd > ? THEN 1 ELSE 0 END), 0) AS over_budget "
                "FROM query_metrics"
            )
            row = self._conn.execute(sql, (COST_BUDGET,)).fetchone()
        return dict(row)

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: MetricsStore | None = None


def _default_db_path() -> str:
    """Resolve the default DB path from settings."""
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / "graph_checkpoints.db")


def get_store(db_path: str | None = None) -> MetricsStore:
    """Return the singleton MetricsStore, creating it on first call."""
    global _store
    if _store is None:
        _store = MetricsStore(db_path or _default_db_path())
        logger.info("MetricsStore initialized at %s", db_path or _default_db_path())
    return _store


def reset_store() -> None:
    """Close and discard the singleton (for testing)."""
    global _store
    if _store is not None:
        _store.close()
        _store = None

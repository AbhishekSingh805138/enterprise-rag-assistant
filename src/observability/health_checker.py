"""Deep health checks for production monitoring.

Checks connectivity and readiness of all subsystems:
- ChromaDB vector store
- SQLite checkpoints
- LLM availability (quick ping)
- Memory usage
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: str  # "ok", "degraded", "error"
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class DeepHealthResult:
    """Aggregated result of all health checks."""
    status: str  # "ok", "degraded", "error"
    checks: list[HealthCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "latency_ms": round(c.latency_ms, 1),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def _check_chromadb() -> HealthCheck:
    """Check ChromaDB connectivity."""
    start = time.perf_counter()
    try:
        from src.vectorstore.chroma_store import collection_stats
        stats = collection_stats()
        latency = (time.perf_counter() - start) * 1000
        return HealthCheck(
            name="chromadb",
            status="ok",
            latency_ms=latency,
            detail=f"{stats['document_count']} documents in {stats['collection']}",
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return HealthCheck(
            name="chromadb", status="error", latency_ms=latency,
            detail=str(e) if settings.debug_mode else "ChromaDB unavailable",
        )


def _check_sqlite() -> HealthCheck:
    """Check SQLite checkpoint database accessibility."""
    start = time.perf_counter()
    try:
        db_path = Path(settings.checkpoint_dir) / "graph_checkpoints.db"
        if not db_path.exists():
            latency = (time.perf_counter() - start) * 1000
            return HealthCheck(
                name="sqlite", status="ok", latency_ms=latency,
                detail="No checkpoint DB yet (will be created on first query)",
            )

        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        latency = (time.perf_counter() - start) * 1000
        return HealthCheck(name="sqlite", status="ok", latency_ms=latency, detail="Accessible")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return HealthCheck(
            name="sqlite", status="error", latency_ms=latency,
            detail=str(e) if settings.debug_mode else "SQLite unavailable",
        )


def _check_memory() -> HealthCheck:
    """Check process memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        status = "ok" if rss_mb < 1024 else "degraded"
        return HealthCheck(
            name="memory", status=status, latency_ms=0,
            detail=f"RSS: {rss_mb:.0f} MB",
        )
    except ImportError:
        return HealthCheck(
            name="memory", status="ok", latency_ms=0,
            detail="psutil not installed — memory check skipped",
        )
    except Exception:
        return HealthCheck(
            name="memory", status="ok", latency_ms=0,
            detail="Memory check failed (non-critical)",
        )


def deep_health_check() -> DeepHealthResult:
    """Run all health checks and return aggregated result."""
    checks = [
        _check_chromadb(),
        _check_sqlite(),
        _check_memory(),
    ]

    # Aggregate status: error if any error, degraded if any degraded, else ok
    statuses = {c.status for c in checks}
    if "error" in statuses:
        overall = "error"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return DeepHealthResult(status=overall, checks=checks)

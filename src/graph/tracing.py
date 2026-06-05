"""Per-node tracing for the CRAG graph.

Wraps graph node functions to log timing, input/output sizes, and metadata.
This lightweight decorator is always-on. LangSmith integration (Phase 6)
works automatically via LangChain callbacks when LANGSMITH_TRACING=true.

Phase 8 additions:
  - Module-level metric accumulator for per-node latency percentiles
  - get_node_metrics() / reset_node_metrics() for observability

Usage:
    from src.graph.tracing import traced

    @traced
    def retrieve(state: dict) -> dict:
        ...
"""
from __future__ import annotations

import functools
import logging
import statistics
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Module-level accumulator: node_name -> list of elapsed_ms values
_node_metrics: dict[str, list[float]] = {}


def get_node_metrics() -> dict[str, dict]:
    """Return per-node latency statistics.

    Returns a dict like:
        {"retrieve": {"count": 10, "p50": 120.0, "p95": 340.0, "p99": 500.0}, ...}
    """
    result = {}
    for name, timings in _node_metrics.items():
        n = len(timings)
        if n == 0:
            continue
        sorted_t = sorted(timings)
        result[name] = {
            "count": n,
            "p50": round(sorted_t[n // 2], 1),
            "p95": round(sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1], 1),
            "p99": round(sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1], 1),
            "mean": round(statistics.mean(timings), 1),
            "last": round(sorted_t[-1], 1),
        }
    return result


def get_last_run_latencies() -> dict[str, float]:
    """Return the most recent latency for each node (for API response)."""
    return {
        name: round(timings[-1], 1)
        for name, timings in _node_metrics.items()
        if timings
    }


def reset_node_metrics() -> None:
    """Clear all accumulated metrics (for testing)."""
    _node_metrics.clear()


def traced(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Decorator that adds timing and size logging to a graph node function."""

    @functools.wraps(fn)
    def wrapper(state: dict) -> dict:
        node_name = fn.__name__
        start = time.perf_counter()

        # Log input summary
        question = state.get("question", "")[:80]
        n_docs_in = len(state.get("documents", []))
        logger.debug(
            "TRACE [%s] START — question=%r, docs_in=%d, retries=%d",
            node_name, question, n_docs_in, state.get("retries", 0),
        )

        result = fn(state)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record metric
        _node_metrics.setdefault(node_name, []).append(elapsed_ms)

        # Log output summary
        n_docs_out = len(result.get("documents", []))
        gen_len = len(result.get("generation", ""))
        extra = ""
        if "relevant" in result:
            extra += f", relevant={result['relevant']}"
        if "critic_passed" in result:
            extra += f", critic_passed={result['critic_passed']}"
        if result.get("claims_removed", 0) > 0:
            extra += f", claims_removed={result['claims_removed']}"

        logger.info(
            "TRACE [%s] %.0fms — docs_out=%d, gen_len=%d%s",
            node_name, elapsed_ms, n_docs_out, gen_len, extra,
        )
        return result

    return wrapper

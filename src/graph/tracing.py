"""Per-node tracing for the CRAG graph.

Wraps graph node functions to log timing, input/output sizes, and metadata.
This lightweight decorator is always-on. LangSmith integration (Phase 6)
works automatically via LangChain callbacks when LANGSMITH_TRACING=true.

Usage:
    from src.graph.tracing import traced

    @traced
    def retrieve(state: dict) -> dict:
        ...
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


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

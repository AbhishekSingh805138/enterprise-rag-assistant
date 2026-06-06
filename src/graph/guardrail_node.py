"""Guardrail check node for the CRAG graph.

Runs input safety checks at the start of the pipeline. If the query
fails guardrails, short-circuits to a safe rejection message.
"""
from __future__ import annotations

import logging

from config import settings
from src.graph.tracing import traced

logger = logging.getLogger(__name__)


@traced
def guardrail_check(state: dict) -> dict:
    """Check the query against input guardrails."""
    if not settings.guardrails_enabled:
        return {"guardrail_passed": True, "guardrail_reason": ""}

    question = state.get("question", "")

    from src.security.guardrails import check_guardrails
    result = check_guardrails(question)

    if not result.safe:
        logger.warning("Guardrail failed: %s", result.reason)
        return {
            "guardrail_passed": False,
            "guardrail_reason": result.reason,
            "generation": f"I cannot process this query. {result.reason}",
        }

    return {"guardrail_passed": True, "guardrail_reason": ""}


def route_after_guardrail(state: dict) -> str:
    """Route based on guardrail check: passed -> continue, failed -> end."""
    if state.get("guardrail_passed", True):
        return "continue"
    return "blocked"

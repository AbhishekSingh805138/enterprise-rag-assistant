"""Early out-of-scope detection for the CRAG graph.

Uses lightweight heuristics (keyword + question type analysis) to detect
clearly off-topic queries before spending tokens on retrieval + grading.
No LLM call — this is a fast, deterministic filter.
"""
from __future__ import annotations

import logging
import re

from src.graph.tracing import traced

logger = logging.getLogger(__name__)

# Enterprise domain keywords — if the query contains at least one,
# it's likely in scope. All lowercase for matching.
_DOMAIN_KEYWORDS = frozenset({
    # HR
    "pto", "vacation", "leave", "onboarding", "offboarding", "employee",
    "handbook", "benefits", "remote work", "work from home", "salary",
    "compensation", "performance review", "hiring", "termination",
    # Legal
    "nda", "contract", "vendor", "compliance", "data protection",
    "gdpr", "privacy", "liability", "intellectual property", "policy",
    "indemnification", "non-disclosure",
    # Engineering
    "api", "code review", "pull request", "deployment", "ci/cd",
    "incident", "runbook", "architecture", "endpoint", "versioning",
    "postmortem", "on-call",
    # Finance
    "procurement", "budget", "expense", "invoice", "revenue",
    "quarterly report", "purchase order", "reimbursement", "fiscal",
    # Security
    "security", "mfa", "multi-factor", "password", "access control",
    "encryption", "firewall", "vulnerability", "phishing", "sso",
    "breach", "threat",
    # Operations
    "business continuity", "disaster recovery", "change management",
    "escalation", "stakeholder", "workflow", "sop",
    # General enterprise
    "company", "organization", "department", "team", "process",
    "guideline", "procedure", "requirement", "approval", "document",
})

# Clearly off-topic patterns (case-insensitive)
_OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(weather|forecast|temperature|celsius|fahrenheit)\b",
    r"\b(sports?|football|basketball|soccer|baseball|cricket)\b",
    r"\b(movie|film|actor|actress|director|netflix)\b",
    r"\b(song|music|album|singer|band|concert)\b",
    r"\b(game|gaming|xbox|playstation|nintendo)\b",
    r"\b(celebrity|gossip|tabloid)\b",
    r"\b(horoscope|zodiac|astrology)\b",
]

_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)


def _is_in_scope(query: str) -> bool:
    """Determine if a query is in-scope for the enterprise corpus.

    Returns True if the query should proceed through the pipeline.
    Returns False only if there's clear evidence the query is off-topic.
    """
    if not query or not query.strip():
        return True  # empty queries handled elsewhere

    lower = query.lower()

    # Check for domain keywords — if any match, it's in scope
    for keyword in _DOMAIN_KEYWORDS:
        if keyword in lower:
            return True

    # Check for clearly off-topic patterns
    if _OFF_TOPIC_RE.search(query):
        return False

    # If no strong signal either way, assume in-scope (err on the side of trying)
    return True


@traced
def scope_check(state: dict) -> dict:
    """Check if the query is in-scope for the enterprise corpus."""
    question = state.get("question", "")
    in_scope = _is_in_scope(question)

    if not in_scope:
        logger.info("Scope check: OUT OF SCOPE — %s", question[:120])
    else:
        logger.debug("Scope check: in scope — %s", question[:80])

    return {"in_scope": in_scope}

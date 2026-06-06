"""Input guardrails for query validation and safety.

Checks for:
1. Prompt injection patterns (common attack vectors)
2. PII detection (SSN, credit card, email in queries)
3. Maximum query length enforcement
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    safe: bool
    reason: str
    sanitized_query: str


# --- Prompt injection patterns ---
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(your|previous)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are\s+(a|an)?", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(have|are)", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(instructions|rules|system)", re.IGNORECASE),
]

# --- PII patterns ---
_SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_CC_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b")


def check_guardrails(query: str) -> GuardrailResult:
    """Run all guardrail checks on the input query.

    Returns a GuardrailResult indicating whether the query is safe to process.
    When guardrails are disabled, always returns safe=True.
    """
    if not settings.guardrails_enabled:
        return GuardrailResult(safe=True, reason="", sanitized_query=query)

    # 1. Max query length
    if len(query) > settings.max_query_length:
        logger.warning("Query exceeds max length: %d > %d", len(query), settings.max_query_length)
        return GuardrailResult(
            safe=False,
            reason=f"Query too long ({len(query)} chars). Maximum: {settings.max_query_length}.",
            sanitized_query=query[:settings.max_query_length],
        )

    # 2. Prompt injection detection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Prompt injection detected: %s", query[:100])
            return GuardrailResult(
                safe=False,
                reason="Query contains potentially unsafe instructions.",
                sanitized_query=query,
            )

    # 3. PII detection
    if settings.pii_detection_enabled:
        pii_found = []
        if _SSN_PATTERN.search(query):
            pii_found.append("SSN")
        if _CC_PATTERN.search(query):
            pii_found.append("credit card number")
        if pii_found:
            logger.warning("PII detected in query: %s", ", ".join(pii_found))
            return GuardrailResult(
                safe=False,
                reason=f"Query contains sensitive information ({', '.join(pii_found)}). "
                       "Please remove personal data before querying.",
                sanitized_query=query,
            )

    return GuardrailResult(safe=True, reason="", sanitized_query=query)

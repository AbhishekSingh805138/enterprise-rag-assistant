"""Output filter for redacting PII in LLM responses.

Scans generated answers for potential PII patterns (SSN, credit card,
email, phone) and replaces them with redacted placeholders.
"""
from __future__ import annotations

import logging
import re

from config import settings

logger = logging.getLogger(__name__)

_REDACTIONS = [
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CC_REDACTED]"),
    (re.compile(r"\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b"), "[PHONE_REDACTED]"),
]


def filter_output(response: str) -> str:
    """Redact PII patterns from the LLM response.

    Only active when pii_detection_enabled=True.
    """
    if not settings.pii_detection_enabled:
        return response

    filtered = response
    redacted_count = 0
    for pattern, replacement in _REDACTIONS:
        new_text = pattern.sub(replacement, filtered)
        if new_text != filtered:
            redacted_count += 1
        filtered = new_text

    if redacted_count > 0:
        logger.info("Output filter redacted %d PII pattern(s)", redacted_count)

    return filtered

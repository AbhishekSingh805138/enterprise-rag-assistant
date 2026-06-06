"""Build a formatted memory context string from conversation history.

Retrieves recent history and formats it as role-tagged messages,
truncating to a token budget to avoid overflowing the LLM context window.
"""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def build_memory_context(
    history: list[dict],
    max_tokens: int | None = None,
) -> str:
    """Format conversation history into a context string for the LLM prompt.

    Args:
        history: List of dicts with 'role' and 'content' keys (oldest first).
        max_tokens: Maximum token budget for the memory context.

    Returns:
        Formatted string like:
            Previous conversation:
            User: What is PTO?
            Assistant: PTO stands for ...
            User: How many days?

        Returns empty string if history is empty or memory is disabled.
    """
    if not settings.memory_enabled or not history:
        return ""

    budget = max_tokens or settings.memory_max_tokens

    # Build from most recent backward, then reverse to maintain chronological order
    lines: list[str] = []
    token_count = 0
    header = "Previous conversation:\n"
    token_count += _estimate_tokens(header)

    for msg in reversed(history):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        label = "User" if role == "user" else "Assistant"
        line = f"{label}: {content}"
        line_tokens = _estimate_tokens(line)

        if token_count + line_tokens > budget:
            break
        lines.append(line)
        token_count += line_tokens

    if not lines:
        return ""

    lines.reverse()
    return header + "\n".join(lines)

"""Tool routing node for the CRAG graph.

Detects queries that need tool assistance (math calculations, department
lookups) and invokes the appropriate tool. Results are stored in
state["tool_results"] so the generation node can include them in context.

Gated behind ENABLE_TOOLS feature flag (default: off).
"""
from __future__ import annotations

import logging
import re

from config import settings
from src.graph.tracing import traced

logger = logging.getLogger(__name__)

# Patterns indicating a math/calculation need
_MATH_PATTERNS = [
    re.compile(r"\b\d+\s*[\+\-\*\/\%\^]\s*\d+", re.IGNORECASE),
    re.compile(r"\bcalculat[e|ion]", re.IGNORECASE),
    re.compile(r"\bhow much is\b", re.IGNORECASE),
    re.compile(r"\bwhat is \d+", re.IGNORECASE),
    re.compile(r"\btotal\s+(of|cost|amount|sum)\b", re.IGNORECASE),
    re.compile(r"\bpercentage\b", re.IGNORECASE),
    re.compile(r"\baverage\b", re.IGNORECASE),
]

# Patterns indicating a department-specific lookup need
_LOOKUP_PATTERNS = [
    re.compile(r"\blook\s*up\b", re.IGNORECASE),
    re.compile(r"\bfind\s+(in|from|the)\b", re.IGNORECASE),
    re.compile(r"\bsearch\s+(in|the|for)\s+\w+\s+department\b", re.IGNORECASE),
    re.compile(r"\b(hr|legal|engineering|finance|security|operations)\s+(policy|policies|document|docs)\b", re.IGNORECASE),
]

# Extract math expressions from a query
_EXPRESSION_RE = re.compile(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/\%\^]\s*\d+(?:\.\d+)?)+)")


def _needs_calculator(query: str) -> bool:
    """Check if the query contains a math expression or calculation request."""
    return any(p.search(query) for p in _MATH_PATTERNS)


def _needs_lookup(query: str) -> bool:
    """Check if the query explicitly requests a department lookup."""
    return any(p.search(query) for p in _LOOKUP_PATTERNS)


def _extract_expression(query: str) -> str | None:
    """Extract the first math expression from a query."""
    match = _EXPRESSION_RE.search(query)
    return match.group(1).strip() if match else None


def _extract_department(query: str) -> str | None:
    """Extract department name from query if mentioned."""
    departments = {"hr", "legal", "engineering", "finance", "security", "operations"}
    q_lower = query.lower()
    for dept in departments:
        if dept in q_lower:
            return dept
    return None


@traced
def tool_router(state: dict) -> dict:
    """Detect tool-needing queries and invoke the appropriate tool.

    Stores results in tool_results for the generation node to use.
    Only runs when ENABLE_TOOLS is True.

    When MCP_ENABLED is True, delegates to the MCP tool router
    for LLM-based tool selection. Falls back to regex routing otherwise.
    """
    if not settings.enable_tools:
        return {"tool_results": []}

    question = state["question"]
    tool_results: list[str] = list(state.get("tool_results", []))

    # MCP-powered routing (LLM-based tool selection)
    if settings.mcp_enabled:
        try:
            from src.mcp.tool_router import mcp_route_and_invoke
            mcp_results = mcp_route_and_invoke(question)
            tool_results.extend(mcp_results)
            if mcp_results:
                logger.info("MCP tool router: %d result(s)", len(mcp_results))
                return {"tool_results": tool_results}
        except Exception:
            logger.debug("MCP routing failed — falling back to regex", exc_info=True)

    # Check for calculator need
    if _needs_calculator(question):
        expression = _extract_expression(question)
        if expression:
            try:
                from src.tools.calculator import calculator
                result = calculator.invoke(expression)
                tool_results.append(f"Calculator({expression}) = {result}")
                logger.info("Tool router: calculator(%s) = %s", expression, result)
            except Exception:
                logger.debug("Calculator invocation failed", exc_info=True)

    # Check for explicit department lookup
    if _needs_lookup(question):
        dept = _extract_department(question)
        try:
            from src.tools.data_lookup import data_lookup
            result = data_lookup.invoke({"query": question, "department": dept or ""})
            tool_results.append(f"DataLookup: {result[:500]}")
            logger.info("Tool router: data_lookup(dept=%s) returned %d chars", dept, len(result))
        except Exception:
            logger.debug("Data lookup invocation failed", exc_info=True)

    if tool_results:
        logger.info("Tool router: %d tool result(s) for query", len(tool_results))

    return {"tool_results": tool_results}

"""Department-filtered document lookup tool.

Retrieves documents from the vector store with optional metadata filtering
by department. Useful for targeted queries like "look up HR policies" or
"find legal contract terms".
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

VALID_DEPARTMENTS = {"hr", "legal", "engineering", "finance", "security", "operations", "general"}


@tool
def data_lookup(query: str, department: str = "") -> str:
    """Look up information from the enterprise document corpus.

    Args:
        query: The search query.
        department: Optional department filter (hr, legal, engineering,
                    finance, security, operations). Leave empty for all.

    Returns:
        Formatted text from the top matching document chunks.
    """
    from src.retrieval import get_retriever

    metadata_filter = None
    if department:
        dept = department.lower().strip()
        if dept not in VALID_DEPARTMENTS:
            return (
                f"Unknown department '{department}'. "
                f"Valid departments: {', '.join(sorted(VALID_DEPARTMENTS))}"
            )
        if dept != "general":
            metadata_filter = {"department": dept}

    try:
        retriever = get_retriever(strategy="dense", k=4, filter=metadata_filter)
        docs = retriever.invoke(query)

        if not docs:
            return f"No documents found for query: {query}"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            dept = doc.metadata.get("department", "unknown")
            results.append(
                f"[{i}] ({source}, {dept})\n{doc.page_content[:500]}"
            )

        formatted = "\n\n".join(results)
        logger.info(
            "Data lookup: %d results for query=%r, department=%r",
            len(docs), query[:80], department,
        )
        return formatted

    except Exception as e:
        logger.exception("Data lookup failed: %s", query[:80])
        return f"Lookup failed: {e}"

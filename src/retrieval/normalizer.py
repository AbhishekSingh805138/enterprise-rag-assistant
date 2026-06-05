"""Query normalization for enterprise RAG retrieval.

Expands common enterprise acronyms, strips trailing punctuation, and
collapses whitespace so retrieval sees a cleaner query.
"""
from __future__ import annotations

import re

# Common enterprise acronyms -> expanded forms
_ACRONYMS: dict[str, str] = {
    "PTO": "paid time off",
    "SLA": "service level agreement",
    "NDA": "non-disclosure agreement",
    "MFA": "multi-factor authentication",
    "WFH": "work from home",
    "KPI": "key performance indicator",
    "OKR": "objectives and key results",
    "RTO": "return to office",
    "BYOD": "bring your own device",
    "SSO": "single sign-on",
    "RBAC": "role-based access control",
    "SOC": "security operations center",
    "BCP": "business continuity plan",
    "DR": "disaster recovery",
    "CI": "continuous integration",
    "CD": "continuous deployment",
    "PR": "pull request",
    "EOD": "end of day",
    "ETA": "estimated time of arrival",
    "RACI": "responsible accountable consulted informed",
    "POC": "point of contact",
    "ROI": "return on investment",
    "CAPEX": "capital expenditure",
    "OPEX": "operational expenditure",
    "PII": "personally identifiable information",
    "GDPR": "general data protection regulation",
    "SOX": "Sarbanes-Oxley",
    "IP": "intellectual property",
}

# Pre-compiled pattern: match whole-word acronyms (case-insensitive)
_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ACRONYMS) + r")\b",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Normalize a query for improved retrieval.

    Steps:
      1. Strip leading/trailing whitespace
      2. Collapse internal whitespace to single spaces
      3. Expand known enterprise acronyms (appends expansion in parentheses)
      4. Remove trailing question marks
    """
    if not query:
        return query

    # Strip and collapse whitespace
    text = re.sub(r"\s+", " ", query.strip())

    # Expand acronyms: "PTO policy" -> "PTO (paid time off) policy"
    def _expand(match: re.Match) -> str:
        acronym = match.group(0).upper()
        expansion = _ACRONYMS.get(acronym, "")
        if expansion:
            return f"{match.group(0)} ({expansion})"
        return match.group(0)

    text = _ACRONYM_PATTERN.sub(_expand, text)

    # Remove trailing question marks (retrieval doesn't benefit from them)
    text = text.rstrip("?").rstrip()

    return text

"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small set of Document objects for unit tests (no API calls needed)."""
    return [
        Document(
            page_content="Employees may work remotely up to 3 days per week with manager approval.",
            metadata={
                "source": "/data/hr/handbook.md",
                "filename": "handbook.md",
                "doc_type": "md",
                "department": "hr",
                "access_level": "internal",
                "start_index": 0,
            },
        ),
        Document(
            page_content="Standard payment terms are Net 30 from the date of invoice.",
            metadata={
                "source": "/data/legal/vendor_contract_terms.md",
                "filename": "vendor_contract_terms.md",
                "doc_type": "md",
                "department": "legal",
                "access_level": "confidential",
                "start_index": 0,
            },
        ),
        Document(
            page_content="All API endpoints must require authentication using OAuth 2.0 with JWT.",
            metadata={
                "source": "/data/engineering/api_guidelines.md",
                "filename": "api_guidelines.md",
                "doc_type": "md",
                "department": "engineering",
                "access_level": "internal",
                "start_index": 0,
            },
        ),
    ]


@pytest.fixture
def sample_docs_path(tmp_path) -> str:
    """Create a temporary directory with sample documents for loader tests."""
    # Create department subdirs
    hr = tmp_path / "hr"
    hr.mkdir()
    (hr / "policy.md").write_text(
        "# Leave Policy\nEmployees get 20 days PTO per year.\n",
        encoding="utf-8",
    )
    (hr / "handbook.txt").write_text(
        "Remote work is allowed 3 days per week.\n",
        encoding="utf-8",
    )

    legal = tmp_path / "legal"
    legal.mkdir()
    (legal / "terms.md").write_text(
        "# Vendor Terms\nPayment terms are Net 30.\nSLA uptime is 99.9%.\n",
        encoding="utf-8",
    )

    # A file at root level (no department subfolder)
    (tmp_path / "readme.md").write_text(
        "# Company Overview\nAcme Corp was founded in 2010.\n",
        encoding="utf-8",
    )

    return str(tmp_path)

"""Load raw documents from a directory into LangChain Document objects.

Supports .pdf, .txt, and .md out of the box. Each loaded Document carries
metadata (source path, file type) so the vector store can do metadata-filtered
retrieval later (e.g. by department or doc_type).
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_path(path: str | Path) -> list[Document]:
    """Load a single file or every supported file in a directory."""
    path = Path(path)
    files = (
        [path]
        if path.is_file()
        else [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
    )

    docs: list[Document] = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(f)).load()
        elif suffix in {".txt", ".md"}:
            loaded = TextLoader(str(f), encoding="utf-8").load()
        else:
            continue

        # Enrich metadata so we can filter at query time.
        for d in loaded:
            d.metadata.setdefault("source", str(f))
            d.metadata["doc_type"] = suffix.lstrip(".")
            d.metadata["filename"] = f.name
        docs.extend(loaded)

    if not docs:
        raise FileNotFoundError(f"No supported documents found under {path!s}")
    return docs

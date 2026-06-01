"""Load raw documents from a directory into LangChain Document objects.

Supports .pdf, .txt, and .md out of the box. Each loaded Document carries
metadata (source path, file type, department, access_level) so the vector
store can do metadata-filtered retrieval later.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Documents in these folders are marked confidential; everything else is internal.
_CONFIDENTIAL_DEPARTMENTS = {"legal", "security"}


def _infer_department(file_path: Path, root: Path) -> str:
    """Derive department from the first subfolder under *root*.

    Example: root/legal/contract.md  →  "legal"
             root/report.md          →  "general"
    """
    try:
        relative = file_path.relative_to(root)
        parts = relative.parts
        if len(parts) > 1:
            return parts[0].lower()
    except ValueError:
        pass
    return "general"


def _infer_access_level(department: str) -> str:
    """Simple rule: confidential departments get 'confidential', rest 'internal'."""
    return "confidential" if department in _CONFIDENTIAL_DEPARTMENTS else "internal"


def load_path(path: str | Path) -> list[Document]:
    """Load a single file or every supported file in a directory.

    Enriches each Document's metadata with:
      - source, filename, doc_type  (original)
      - department, access_level    (new — inferred from folder structure)
    """
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent

    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    )

    if not files:
        raise FileNotFoundError(f"No supported documents found under {path!s}")

    docs: list[Document] = []
    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(f)).load()
            elif suffix in {".txt", ".md"}:
                loaded = TextLoader(str(f), encoding="utf-8").load()
            else:
                continue
        except Exception:
            logger.exception("Failed to load %s — skipping", f)
            continue

        department = _infer_department(f, root)
        access_level = _infer_access_level(department)

        for d in loaded:
            d.metadata.setdefault("source", str(f))
            d.metadata["doc_type"] = suffix.lstrip(".")
            d.metadata["filename"] = f.name
            d.metadata["department"] = department
            d.metadata["access_level"] = access_level
        docs.extend(loaded)

        logger.info(
            "Loaded %s (%d page(s), dept=%s, access=%s)",
            f.name, len(loaded), department, access_level,
        )

    if not docs:
        raise FileNotFoundError(f"No supported documents found under {path!s}")

    logger.info("Total documents loaded: %d from %d file(s)", len(docs), len(files))
    return docs

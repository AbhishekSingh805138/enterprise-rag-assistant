"""Ask a question against the ingested corpus.

Usage:
    python -m scripts.ask "What is the remote work policy?"
    python -m scripts.ask --mode graph "What is the remote work policy?"
    python -m scripts.ask --filter department=legal "What are the payment terms?"
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import settings, setup_logging

logger = logging.getLogger(__name__)


def _parse_filter(value: str) -> dict:
    """Parse 'key=value' into a dict for metadata filtering."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Filter must be key=value, got: {value!r}"
        )
    k, v = value.split("=", 1)
    return {k.strip(): v.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question over the ingested document corpus.",
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="The question to ask",
    )
    parser.add_argument(
        "--mode",
        choices=["naive", "graph"],
        default="naive",
        help="Pipeline mode: 'naive' (LCEL baseline) or 'graph' (CRAG LangGraph) (default: naive)",
    )
    parser.add_argument(
        "--filter",
        type=_parse_filter,
        default=None,
        help="Metadata filter as key=value (e.g. department=legal)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help=f"Number of documents to retrieve (default: {settings.top_k})",
    )
    args = parser.parse_args()

    setup_logging()
    settings.validate()

    question = " ".join(args.question)
    if not question.strip():
        print("Error: question cannot be empty.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.mode == "graph":
            from src.graph.build_graph import ask
            print(ask(question))
        else:
            from src.rag.naive_rag import answer
            print(answer(question, k=args.top_k, filter=args.filter))
    except Exception as e:
        logger.exception("Query failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

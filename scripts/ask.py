"""Ask a question against the ingested corpus.

Usage:
    python -m scripts.ask "What is the remote work policy?"
    python -m scripts.ask --graph "What is the remote work policy?"

--graph uses the Corrective RAG LangGraph pipeline; default uses the
Phase 1 naive baseline.
"""
from __future__ import annotations

import sys

from config import settings


def main() -> None:
    settings.validate()
    args = sys.argv[1:]
    use_graph = "--graph" in args
    args = [a for a in args if a != "--graph"]

    if not args:
        print('Provide a question, e.g. python -m scripts.ask "..."')
        sys.exit(1)

    question = " ".join(args)

    if use_graph:
        from src.graph.build_graph import ask
        print(ask(question))
    else:
        from src.rag.naive_rag import answer
        print(answer(question))


if __name__ == "__main__":
    main()

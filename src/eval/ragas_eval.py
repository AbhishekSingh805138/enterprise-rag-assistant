"""Phase 2: evaluation harness with RAGAS.

This is the single most important file in the project. It turns "the answers
feel better" into numbers you can defend. Run it against the naive baseline
first, record the scores, then re-run after every retrieval change.

Metrics:
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : are the retrieved chunks actually relevant?
  - context_recall    : did we retrieve everything needed? (needs ground truth)

Build a small eval set (20-50 Q/A pairs over your corpus) and keep it in
version control. Quality of the eval set caps the quality of your conclusions.
"""
from __future__ import annotations

from collections.abc import Callable


# A tiny example eval set. Replace with real Q/A pairs grounded in YOUR docs.
EVAL_SET: list[dict] = [
    {
        "question": "What is the company's remote work policy?",
        "ground_truth": "Employees may work remotely up to 3 days per week with manager approval.",
    },
    # ... add 20-50 more
]


def evaluate(answer_fn: Callable[[str], str], retriever) -> dict:
    """Run RAGAS over EVAL_SET.

    answer_fn: a callable taking a question and returning the answer string
               (e.g. src.rag.naive_rag.answer or src.graph.build_graph.ask).
    retriever: used to capture the contexts each question retrieved.
    """
    # Imported lazily so the rest of the project runs without ragas installed.
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in EVAL_SET:
        q = item["question"]
        docs = retriever.invoke(q)
        rows["question"].append(q)
        rows["answer"].append(answer_fn(q))
        rows["contexts"].append([d.page_content for d in docs])
        rows["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(rows)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)


if __name__ == "__main__":
    from src.rag.naive_rag import answer
    from src.vectorstore.chroma_store import get_retriever

    scores = evaluate(answer, get_retriever())
    print("\n=== Baseline RAGAS scores ===")
    for metric, value in scores.items():
        print(f"{metric:>20}: {value:.3f}")

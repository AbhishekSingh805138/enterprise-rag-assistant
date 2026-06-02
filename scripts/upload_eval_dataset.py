"""Upload the evaluation dataset to LangSmith.

One-time script that pushes the 60 Q/A evaluation pairs from
src/eval/eval_set.json to LangSmith as a named dataset for
tracking eval runs over time.

Usage:
    python -m scripts.upload_eval_dataset                          # upload
    python -m scripts.upload_eval_dataset --name my-dataset        # custom name
    python -m scripts.upload_eval_dataset --dry-run                # preview only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
EVAL_SET_PATH = PROJECT_ROOT / "src" / "eval" / "eval_set.json"
DEFAULT_DATASET_NAME = "enterprise-rag-eval"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload evaluation dataset to LangSmith",
    )
    parser.add_argument(
        "--name", type=str, default=DEFAULT_DATASET_NAME,
        help=f"Dataset name in LangSmith (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would be uploaded without calling the API",
    )
    args = parser.parse_args()

    # Load eval set
    if not EVAL_SET_PATH.exists():
        print(f"Error: Evaluation set not found at {EVAL_SET_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluation items from {EVAL_SET_PATH}")

    if args.dry_run:
        print(f"[DRY RUN] Would create dataset '{args.name}' with {len(eval_set)} examples")
        print(f"[DRY RUN] Sample item: {eval_set[0]['question'][:80]}...")
        return

    # Check for API key
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        print(
            "Error: LANGSMITH_API_KEY is not set. "
            "Set it in your .env or environment to upload.",
            file=sys.stderr,
        )
        sys.exit(1)

    from langsmith import Client

    client = Client()

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=args.name)
        print(f"Dataset '{args.name}' already exists (id={existing.id}). Skipping creation.")
        print("Use a different --name to create a new dataset.")
        return
    except Exception:
        pass  # Dataset doesn't exist — proceed with creation

    # Create dataset and examples
    dataset = client.create_dataset(
        dataset_name=args.name,
        description=f"Enterprise RAG evaluation set ({len(eval_set)} Q/A pairs)",
    )
    print(f"Created dataset '{args.name}' (id={dataset.id})")

    client.create_examples(
        inputs=[{"question": item["question"]} for item in eval_set],
        outputs=[{"ground_truth": item["ground_truth"]} for item in eval_set],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(eval_set)} examples to LangSmith dataset '{args.name}'")


if __name__ == "__main__":
    main()

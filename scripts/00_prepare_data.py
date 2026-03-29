#!/usr/bin/env python3
"""
Step 0: 데이터 준비.

지원 데이터셋:
  - nq: Natural Questions (full) — 기본값, Lee et al. NQ-Open의 full 버전
  - triviaqa: TriviaQA rc subset

Usage:
    # NQ (full) — 기본
    python scripts/00_prepare_data.py

    # TriviaQA rc
    python scripts/00_prepare_data.py --dataset triviaqa

    # NQ, context 길이 제한
    python scripts/00_prepare_data.py --max-context-length 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare QA dataset")
    parser.add_argument(
        "--dataset",
        choices=["nq", "triviaqa"],
        default="triviaqa",
        help="Dataset to prepare (default: nq)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=3000,
        help="Max context length in characters (default: 3000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: auto-generated)",
    )
    args = parser.parse_args()

    # Auto output path
    if args.output is None:
        if args.dataset == "nq":
            args.output = "data/processed/nq_full_processed.jsonl"
        else:
            args.output = "data/processed/triviaqa_rc_evidence_present.jsonl"

    logger.info("=" * 60)
    logger.info(f"Step 0: Data Preparation ({args.dataset})")
    logger.info("=" * 60)

    if args.dataset == "nq":
        from src.data.nq_loader import load_nq_full, save_processed_data

        data = load_nq_full(
            split=args.split,
            max_context_length=args.max_context_length,
        )
    elif args.dataset == "triviaqa":
        from src.data.triviaqa_loader import load_triviaqa_rc, save_processed_data

        data = load_triviaqa_rc(
            split=args.split,
            max_context_length=args.max_context_length,
        )
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    if not data:
        logger.error(
            "No data loaded. Check your internet connection and HuggingFace access."
        )
        sys.exit(1)

    save_processed_data(data, args.output)
    logger.info(f"Done! {len(data)} items saved to {args.output}")


if __name__ == "__main__":
    main()

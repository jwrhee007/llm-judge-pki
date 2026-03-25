#!/usr/bin/env python3
"""
Step 0: TriviaQA rc 데이터 준비.

1. HuggingFace datasets에서 TriviaQA rc validation split 로드
2. evidence_present 필터링 (answer alias가 context에 포함된 문항만)
3. 결과를 data/processed/triviaqa_rc_evidence_present.jsonl로 저장

Usage:
    python scripts/00_prepare_data.py
    python scripts/00_prepare_data.py --max-context-length 2000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.triviaqa_loader import load_triviaqa_rc, save_processed_data
from src.utils.logger import setup_logger

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare TriviaQA rc data")
    parser.add_argument(
        "--split", default="validation", help="Dataset split (default: validation)"
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=3000,
        help="Max context length in characters (default: 3000)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/triviaqa_rc_evidence_present.jsonl",
        help="Output file path",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 0: TriviaQA rc Data Preparation")
    logger.info("=" * 60)

    # Load and filter
    data = load_triviaqa_rc(
        split=args.split,
        max_context_length=args.max_context_length,
    )

    if not data:
        logger.error("No data loaded. Check your internet connection and HuggingFace access.")
        sys.exit(1)

    # Save
    save_processed_data(data, args.output)

    logger.info(f"Done! {len(data)} evidence-present items saved to {args.output}")


if __name__ == "__main__":
    main()

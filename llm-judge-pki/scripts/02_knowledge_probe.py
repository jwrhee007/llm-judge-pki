#!/usr/bin/env python3
"""
Step 2: Knowledge Probe 실행 (Exp. 2-0).

Judge에게 context 없이 질문만 제시하여 사전지식 보유 여부를 확인한다.
방식 A (Bare question) × 3회 + 방식 B (Knowledge-eliciting) × 3회.

실행 모드:
  - sync:  동기 방식 (소규모 테스트 / 디버깅용)
  - batch: OpenAI Batch API (대규모 실행 — 권장)

Usage:
    # 동기 방식 (전체)
    python scripts/02_knowledge_probe.py

    # Batch API 방식 (권장)
    python scripts/02_knowledge_probe.py --use-batch

    # 특정 방식만 실행
    python scripts/02_knowledge_probe.py --methods A
    python scripts/02_knowledge_probe.py --methods B
    python scripts/02_knowledge_probe.py --methods A B

    # Batch submit만 (결과 수집은 나중에)
    python scripts/02_knowledge_probe.py --use-batch --batch-submit-only

    # Batch 결과 수집 (이전에 submit한 batch)
    python scripts/02_knowledge_probe.py --use-batch --batch-collect \
        --batch-id-a batch_xxx --batch-id-b batch_yyy

    # 스모크 테스트 (5문항만)
    python scripts/02_knowledge_probe.py --smoke-test
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.openai_client import OpenAIClient
from src.data.sampler import load_sampled_data
from src.probes.knowledge_probe import (
    classify_from_batch,
    compute_probe_summary,
    parse_probe_batch_results,
    prepare_probe_batch_requests,
    run_probe_all,
    save_probe_results,
)
from src.utils.logger import setup_logger

logger = setup_logger("knowledge_probe")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_summary(summary: dict, method: str) -> None:
    """Print a summary of probe results."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Knowledge Probe Summary — Method {method}")
    logger.info(f"{'='*50}")
    logger.info(f"Total items: {summary['total']}")

    overall = summary["overall"]
    for level in ["strong-knows", "weak-knows", "guess", "doesn't-know"]:
        count = overall.get(level, 0)
        pct = count / summary["total"] * 100 if summary["total"] > 0 else 0
        logger.info(f"  {level:15s}: {count:4d}  ({pct:5.1f}%)")

    logger.info(f"\nBy NER tag:")
    for tag, counts in sorted(summary["by_ner_tag"].items()):
        total_tag = sum(counts.values())
        parts = []
        for level in ["strong-knows", "weak-knows", "guess", "doesn't-know"]:
            c = counts.get(level, 0)
            parts.append(f"{level[:2]}={c}")
        logger.info(f"  {tag:15s}: {total_tag:3d}  ({', '.join(parts)})")


# =====================================================================
# Synchronous execution
# =====================================================================

def run_sync(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    methods: list[str],
    output_dir: str,
) -> None:
    """Run Knowledge Probe synchronously for each method."""
    model = config["model"]["judge"]
    n_trials = config["probe"]["n_trials"]
    temperature = config["probe"]["temperature"]
    max_tokens = config["model"]["max_tokens_probe"]
    match_model = config["model"]["probe_answer_checker"]

    for method in methods:
        logger.info(f"\n--- Running Probe (method={method}, sync) ---")
        logger.info(
            f"Model: {model}, Trials: {n_trials}, "
            f"Temperature: {temperature}, Items: {len(data)}"
        )

        results = run_probe_all(
            client=client,
            data=data,
            method=method,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
            match_model=match_model,
        )

        # Save
        output_path = f"{output_dir}/probe_method_{method}.jsonl"
        save_probe_results(results, output_path)

        # Summary
        summary = compute_probe_summary(results)
        log_summary(summary, method)

        # Save summary
        summary_path = f"{output_dir}/probe_method_{method}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


# =====================================================================
# Batch API execution
# =====================================================================

def run_batch_submit(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    methods: list[str],
    batch_output_dir: str,
) -> dict[str, str]:
    """Submit batch jobs for each method. Returns {method: batch_id}."""
    model = config["model"]["judge"]
    n_trials = config["probe"]["n_trials"]
    temperature = config["probe"]["temperature"]
    max_tokens = config["model"]["max_tokens_probe"]

    batch_ids = {}
    for method in methods:
        logger.info(f"\n--- Preparing Batch (method={method}) ---")

        requests, _idx_map = prepare_probe_batch_requests(
            data=data,
            method=method,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        batch_file = client.create_batch_file(
            requests=requests,
            output_path=f"{batch_output_dir}/probe_{method}_input.jsonl",
        )

        batch_id = client.submit_batch(
            batch_file,
            description=f"Knowledge Probe method={method}",
        )
        batch_ids[method] = batch_id
        logger.info(f"Method {method}: batch_id = {batch_id}")

    # Save batch IDs for later collection
    ids_path = f"{batch_output_dir}/probe_batch_ids.json"
    with open(ids_path, "w") as f:
        json.dump(batch_ids, f, indent=2)
    logger.info(f"Batch IDs saved to {ids_path}")

    return batch_ids


def run_batch_collect(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    batch_ids: dict[str, str],
    batch_output_dir: str,
    output_dir: str,
) -> None:
    """Poll, download, and process batch results for each method."""
    n_trials = config["probe"]["n_trials"]
    match_model = config["model"]["probe_answer_checker"]

    for method, batch_id in batch_ids.items():
        logger.info(f"\n--- Collecting Batch (method={method}, id={batch_id}) ---")

        # Poll
        batch = client.poll_batch(batch_id, poll_interval=30)
        if batch.status != "completed":
            logger.error(f"Batch {batch_id} status: {batch.status}")
            continue

        # Download
        raw_results = client.download_batch_results(
            batch_id=batch_id,
            output_path=f"{batch_output_dir}/probe_{method}_output.jsonl",
        )

        # Parse
        grouped = parse_probe_batch_results(
            batch_results=raw_results,
            data=data,
            method=method,
            n_trials=n_trials,
        )

        # Answer matching + classification
        results = classify_from_batch(
            grouped_results=grouped,
            data=data,
            client=client,
            method=method,
            match_model=match_model,
            n_trials=n_trials,
        )

        # Save
        output_path = f"{output_dir}/probe_method_{method}.jsonl"
        save_probe_results(results, output_path)

        # Summary
        summary = compute_probe_summary(results)
        log_summary(summary, method)

        summary_path = f"{output_dir}/probe_method_{method}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


def run_batch_full(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    methods: list[str],
    batch_output_dir: str,
    output_dir: str,
) -> None:
    """Submit → poll → collect in one go."""
    batch_ids = run_batch_submit(
        client=client,
        data=data,
        config=config,
        methods=methods,
        batch_output_dir=batch_output_dir,
    )
    run_batch_collect(
        client=client,
        data=data,
        config=config,
        batch_ids=batch_ids,
        batch_output_dir=batch_output_dir,
        output_dir=output_dir,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp. 2-0: Knowledge Probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_sampled.jsonl",
        help="Input sampled data file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["A", "B"],
        default=["A", "B"],
        help="Probe methods to run (default: A B)",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use OpenAI Batch API",
    )
    parser.add_argument(
        "--batch-submit-only",
        action="store_true",
        help="Only submit batch jobs (don't wait for results)",
    )
    parser.add_argument(
        "--batch-collect",
        action="store_true",
        help="Collect results from previously submitted batches",
    )
    parser.add_argument(
        "--batch-id-a",
        default=None,
        help="Batch ID for method A (with --batch-collect)",
    )
    parser.add_argument(
        "--batch-id-b",
        default=None,
        help="Batch ID for method B (with --batch-collect)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with only 5 items for testing",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    output_dir = args.output_dir or config["probe"]["output_dir"]
    batch_output_dir = config["api"]["batch_output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(batch_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Exp. 2-0: Knowledge Probe")
    logger.info("=" * 60)

    # Load data
    data = load_sampled_data(args.input)

    if args.smoke_test:
        data = data[:5]
        logger.info(f"Smoke test mode: using {len(data)} items")

    # Initialize client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAIClient(
        api_key=api_key,
        max_retries=config["api"]["max_retries"],
        requests_per_minute=config["api"]["requests_per_minute"],
    )

    logger.info(f"Judge model: {config['model']['judge']}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Trials per method: {config['probe']['n_trials']}")
    logger.info(f"Temperature: {config['probe']['temperature']}")
    logger.info(f"Total items: {len(data)}")
    expected_calls = len(data) * config["probe"]["n_trials"] * len(args.methods)
    logger.info(f"Expected API calls (probe): {expected_calls}")

    # Execute
    if args.batch_collect:
        # Collect from previously submitted batches
        batch_ids = {}
        if "A" in args.methods and args.batch_id_a:
            batch_ids["A"] = args.batch_id_a
        if "B" in args.methods and args.batch_id_b:
            batch_ids["B"] = args.batch_id_b

        if not batch_ids:
            # Try loading from saved file
            ids_path = f"{batch_output_dir}/probe_batch_ids.json"
            if Path(ids_path).exists():
                with open(ids_path) as f:
                    saved_ids = json.load(f)
                for m in args.methods:
                    if m in saved_ids:
                        batch_ids[m] = saved_ids[m]
                logger.info(f"Loaded batch IDs from {ids_path}: {batch_ids}")
            else:
                logger.error(
                    "No batch IDs provided. Use --batch-id-a / --batch-id-b "
                    "or run --batch-submit-only first."
                )
                sys.exit(1)

        run_batch_collect(
            client=client,
            data=data,
            config=config,
            batch_ids=batch_ids,
            batch_output_dir=batch_output_dir,
            output_dir=output_dir,
        )

    elif args.use_batch:
        if args.batch_submit_only:
            run_batch_submit(
                client=client,
                data=data,
                config=config,
                methods=args.methods,
                batch_output_dir=batch_output_dir,
            )
            logger.info(
                "Batch jobs submitted. Run with --batch-collect to retrieve results."
            )
        else:
            run_batch_full(
                client=client,
                data=data,
                config=config,
                methods=args.methods,
                batch_output_dir=batch_output_dir,
                output_dir=output_dir,
            )

    else:
        # Synchronous execution
        run_sync(
            client=client,
            data=data,
            config=config,
            methods=args.methods,
            output_dir=output_dir,
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

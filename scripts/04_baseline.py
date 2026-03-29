#!/usr/bin/env python3
"""
Step 4: Exp. 2-1 Baseline — ctx_orig + correct answer.

원본 context와 correct answer를 사용하여 ACC_orig를 산출한다.
P-Lee-Standard 프롬프트, T=0, 30회 반복.

Batch를 chunk 단위로 나눠서 순차 실행하여
enqueued token limit 문제를 회피한다.

Usage:
    # Batch API — chunked (권장)
    python scripts/04_baseline.py --use-batch

    # 동기 방식
    python scripts/04_baseline.py

    # 스모크 테스트
    python scripts/04_baseline.py --smoke-test

    # chunk 크기 조절
    python scripts/04_baseline.py --use-batch --chunk-size 3000

    # Batch submit만 (단일 chunk, 디버깅용)
    python scripts/04_baseline.py --use-batch --batch-submit-only

    # Batch 결과 수집 (이전 submit)
    python scripts/04_baseline.py --use-batch --batch-collect --batch-id batch_xxx
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
from src.evaluation.judge_runner import (
    classify_judge_batch,
    compute_judge_summary,
    parse_judge_batch_results,
    prepare_judge_batch_requests,
    run_judge_all,
    save_judge_results,
)
from src.utils.logger import setup_logger

logger = setup_logger("baseline")

EXPERIMENT_TAG = "exp2_1_baseline"
PROMPT_ID = "P-Lee-Standard"
GROUND_TRUTH = "CORRECT"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_summary(summary: dict) -> None:
    """Print baseline summary."""
    logger.info(f"\n{'='*50}")
    logger.info("Exp. 2-1 Baseline Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total items: {summary['total']}")
    logger.info(
        f"ACC_orig (majority=CORRECT): {summary['accuracy']:.4f} "
        f"({summary['correct_count']}/{summary['total']})"
    )
    logger.info(f"Majority verdict distribution: {summary['majority_distribution']}")
    logger.info(f"Mean verdict entropy: {summary['mean_entropy']:.4f}")
    logger.info(
        f"Unstable items (H>0): {summary['unstable_count']} "
        f"({summary['unstable_ratio']:.1%})"
    )

    logger.info(f"\nBy NER/analysis tag:")
    for tag, stats in sorted(
        summary["by_tag"].items(), key=lambda x: -x[1]["total"]
    ):
        logger.info(
            f"  {tag:15s}: n={stats['total']:4d}, "
            f"ACC={stats['accuracy']:.3f}, "
            f"H_mean={stats['mean_entropy']:.4f}, "
            f"unstable={stats['unstable_count']}"
        )


def process_and_save(
    raw_results: list[dict],
    data: list[dict],
    config: dict,
    output_dir: str,
) -> None:
    """Parse batch results, compute summary, and save."""
    n_trials = config.get("judge", {}).get("n_trials", 30)

    # Parse
    grouped = parse_judge_batch_results(
        batch_results=raw_results,
        experiment_tag=EXPERIMENT_TAG,
    )

    # Classify
    results = classify_judge_batch(
        grouped_results=grouped,
        data=data,
        prompt_id=PROMPT_ID,
        n_trials=n_trials,
    )

    # Save
    output_path = f"{output_dir}/baseline_results.jsonl"
    save_judge_results(results, output_path)

    # Summary
    summary = compute_judge_summary(results, ground_truth=GROUND_TRUTH)
    log_summary(summary)

    summary_path = f"{output_dir}/baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")


# =====================================================================
# Synchronous execution
# =====================================================================

def run_sync(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    output_dir: str,
) -> None:
    """Run baseline evaluation synchronously."""
    model = config["model"]["judge"]
    n_trials = config.get("judge", {}).get("n_trials", 30)
    temperature = config["model"]["temperature"]
    max_tokens = config["model"].get("max_tokens_judge", 16)
    seed = config["model"]["seed"]

    logger.info(f"\n--- Running Baseline (sync) ---")
    logger.info(
        f"Model: {model}, Prompt: {PROMPT_ID}, Trials: {n_trials}, "
        f"T={temperature}, Items: {len(data)}"
    )

    results = run_judge_all(
        client=client,
        data=data,
        prompt_id=PROMPT_ID,
        model=model,
        n_trials=n_trials,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    # Save
    output_path = f"{output_dir}/baseline_results.jsonl"
    save_judge_results(results, output_path)

    # Summary
    summary = compute_judge_summary(results, ground_truth=GROUND_TRUTH)
    log_summary(summary)

    summary_path = f"{output_dir}/baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")


# =====================================================================
# Batch API: single batch (submit / collect)
# =====================================================================

def run_batch_submit(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    batch_output_dir: str,
) -> str:
    """Submit a single batch job. Returns batch_id."""
    model = config["model"]["judge"]
    n_trials = config.get("judge", {}).get("n_trials", 30)
    temperature = config["model"]["temperature"]
    max_tokens = config["model"].get("max_tokens_judge", 16)
    seed = config["model"]["seed"]

    logger.info(f"\n--- Preparing Batch (Baseline) ---")

    requests = prepare_judge_batch_requests(
        data=data,
        prompt_id=PROMPT_ID,
        model=model,
        n_trials=n_trials,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        experiment_tag=EXPERIMENT_TAG,
    )

    batch_file = client.create_batch_file(
        requests=requests,
        output_path=f"{batch_output_dir}/{EXPERIMENT_TAG}_input.jsonl",
    )

    batch_id = client.submit_batch(
        batch_file,
        description=(
            f"Exp. 2-1 Baseline ({PROMPT_ID}, "
            f"{len(data)} items × {n_trials} trials)"
        ),
    )
    logger.info(f"Batch submitted: {batch_id}")

    # Save batch ID
    ids_path = f"{batch_output_dir}/{EXPERIMENT_TAG}_batch_id.json"
    with open(ids_path, "w") as f:
        json.dump(
            {"batch_id": batch_id, "experiment": EXPERIMENT_TAG},
            f, indent=2,
        )
    logger.info(f"Batch ID saved to {ids_path}")

    return batch_id


def run_batch_collect(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    batch_id: str,
    batch_output_dir: str,
    output_dir: str,
) -> None:
    """Poll, download, and process a single batch."""
    logger.info(
        f"\n--- Collecting Batch ({EXPERIMENT_TAG}, id={batch_id}) ---"
    )

    # Poll
    batch = client.poll_batch(batch_id, poll_interval=30)
    if batch.status != "completed":
        logger.error(f"Batch {batch_id} status: {batch.status}")
        return

    # Download
    raw_results = client.download_batch_results(
        batch_id=batch_id,
        output_path=f"{batch_output_dir}/{EXPERIMENT_TAG}_output.jsonl",
    )

    process_and_save(raw_results, data, config, output_dir)


# =====================================================================
# Batch API: chunked execution (토큰 리밋 회피)
# =====================================================================

def run_batch_chunked(
    client: OpenAIClient,
    data: list[dict],
    config: dict,
    batch_output_dir: str,
    output_dir: str,
    chunk_size: int = 5000,
) -> None:
    """
    Batch를 chunk_size 단위로 분할하여 순차 실행한다.

    하나의 chunk가 완료(poll)된 후 다음 chunk를 제출하므로
    enqueued token limit 문제를 회피한다.

    중간에 실패하거나 중단되면 진행 상황이 저장되고,
    다시 실행 시 완료된 chunk를 건너뛰고 이어서 진행한다.

    chunk_size는 request 수 기준이다. (문항 × 30회 = requests)
    예: chunk_size=5000이면 ~166문항 × 30회 분량씩 처리.
    """
    model = config["model"]["judge"]
    n_trials = config.get("judge", {}).get("n_trials", 30)
    temperature = config["model"]["temperature"]
    max_tokens = config["model"].get("max_tokens_judge", 16)
    seed = config["model"]["seed"]

    # 전체 requests 생성
    all_requests = prepare_judge_batch_requests(
        data=data,
        prompt_id=PROMPT_ID,
        model=model,
        n_trials=n_trials,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        experiment_tag=EXPERIMENT_TAG,
    )

    total = len(all_requests)
    n_chunks = (total + chunk_size - 1) // chunk_size

    logger.info(
        f"\n--- Chunked Batch Execution ---\n"
        f"  Total requests: {total}\n"
        f"  Chunk size: {chunk_size}\n"
        f"  Number of chunks: {n_chunks}"
    )

    # 이전 진행 상황 로드 (resume 지원)
    progress = _load_chunk_progress(batch_output_dir, EXPERIMENT_TAG)
    completed_chunks: dict[int, str] = {}  # chunk_idx → batch_id

    if progress:
        for entry in progress.get("chunks", []):
            if entry.get("status") == "completed":
                completed_chunks[entry["chunk_idx"]] = entry["batch_id"]
        if completed_chunks:
            logger.info(
                f"Resuming: {len(completed_chunks)}/{n_chunks} chunks "
                f"already completed"
            )

    # Chunk별 순차 실행
    all_raw_results = []
    chunk_records = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)

        # 이미 완료된 chunk → 결과 다운로드만
        if chunk_idx in completed_chunks:
            logger.info(
                f"\n--- Chunk {chunk_idx + 1}/{n_chunks} "
                f"(already completed, downloading results) ---"
            )
            output_path = (
                f"{batch_output_dir}/"
                f"{EXPERIMENT_TAG}_chunk{chunk_idx}_output.jsonl"
            )

            if Path(output_path).exists():
                # 이미 다운로드된 결과 로드
                with open(output_path, "r") as f:
                    chunk_results = [json.loads(line) for line in f]
                logger.info(
                    f"Loaded {len(chunk_results)} cached results from {output_path}"
                )
            else:
                # 다운로드 필요
                batch_id = completed_chunks[chunk_idx]
                chunk_results = client.download_batch_results(
                    batch_id=batch_id,
                    output_path=output_path,
                )

            all_raw_results.extend(chunk_results)
            chunk_records.append({
                "chunk_idx": chunk_idx,
                "batch_id": completed_chunks[chunk_idx],
                "status": "completed",
                "n_requests": end - start,
            })
            continue

        # 새로운 chunk 실행
        chunk_requests = all_requests[start:end]

        logger.info(
            f"\n--- Chunk {chunk_idx + 1}/{n_chunks} "
            f"(requests {start}~{end - 1}, n={len(chunk_requests)}) ---"
        )

        # 1. Write batch file
        chunk_file = client.create_batch_file(
            requests=chunk_requests,
            output_path=(
                f"{batch_output_dir}/"
                f"{EXPERIMENT_TAG}_chunk{chunk_idx}_input.jsonl"
            ),
        )

        # 2. Submit
        batch_id = client.submit_batch(
            chunk_file,
            description=(
                f"{EXPERIMENT_TAG} chunk {chunk_idx + 1}/{n_chunks} "
                f"({len(chunk_requests)} requests)"
            ),
        )
        logger.info(f"Chunk {chunk_idx + 1} submitted: {batch_id}")

        # 3. Poll until complete
        batch = client.poll_batch(batch_id, poll_interval=30)
        if batch.status != "completed":
            logger.error(
                f"Chunk {chunk_idx + 1} failed: {batch.status}. "
                f"Stopping. Re-run to resume from this chunk."
            )
            chunk_records.append({
                "chunk_idx": chunk_idx,
                "batch_id": batch_id,
                "status": batch.status,
                "n_requests": len(chunk_requests),
            })
            _save_chunk_progress(
                chunk_records, n_chunks, batch_output_dir, EXPERIMENT_TAG
            )
            return

        # 4. Download results
        chunk_results = client.download_batch_results(
            batch_id=batch_id,
            output_path=(
                f"{batch_output_dir}/"
                f"{EXPERIMENT_TAG}_chunk{chunk_idx}_output.jsonl"
            ),
        )
        all_raw_results.extend(chunk_results)

        chunk_records.append({
            "chunk_idx": chunk_idx,
            "batch_id": batch_id,
            "status": "completed",
            "n_requests": len(chunk_requests),
        })

        # 매 chunk 완료 시 진행 상황 저장
        _save_chunk_progress(
            chunk_records, n_chunks, batch_output_dir, EXPERIMENT_TAG
        )

        logger.info(
            f"Chunk {chunk_idx + 1}/{n_chunks} complete. "
            f"Cumulative results: {len(all_raw_results)}"
        )

    # All chunks done — process combined results
    logger.info(
        f"\nAll {n_chunks} chunks completed. "
        f"Total results: {len(all_raw_results)}"
    )
    process_and_save(all_raw_results, data, config, output_dir)


def _save_chunk_progress(
    chunk_records: list[dict],
    n_chunks_total: int,
    batch_output_dir: str,
    experiment_tag: str,
) -> None:
    """Save chunk progress for resume."""
    progress_path = f"{batch_output_dir}/{experiment_tag}_chunks.json"
    with open(progress_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_tag,
                "n_chunks_total": n_chunks_total,
                "n_completed": sum(
                    1 for c in chunk_records if c["status"] == "completed"
                ),
                "chunks": chunk_records,
            },
            f, indent=2,
        )
    logger.info(f"Chunk progress saved to {progress_path}")


def _load_chunk_progress(
    batch_output_dir: str,
    experiment_tag: str,
) -> dict | None:
    """Load previous chunk progress if exists."""
    progress_path = f"{batch_output_dir}/{experiment_tag}_chunks.json"
    if not Path(progress_path).exists():
        return None
    with open(progress_path, "r") as f:
        progress = json.load(f)
    logger.info(f"Loaded chunk progress from {progress_path}")
    return progress


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp. 2-1: Baseline (ctx_orig + correct answer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_sampled.jsonl",
        help="Input sampled data file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Output directory",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use OpenAI Batch API",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Batch chunk size in requests (default: from config, fallback 5000)",
    )
    parser.add_argument(
        "--batch-submit-only",
        action="store_true",
        help="Only submit batch job (single, no chunking)",
    )
    parser.add_argument(
        "--batch-collect",
        action="store_true",
        help="Collect results from previously submitted batch",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch ID (with --batch-collect)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with only 5 items",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    output_dir = args.output_dir
    batch_output_dir = config["api"]["batch_output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(batch_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Exp. 2-1: Baseline (ctx_orig + correct answer)")
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

    n_trials = config.get("judge", {}).get("n_trials", 30)
    logger.info(f"Judge model: {config['model']['judge']}")
    logger.info(f"Prompt: {PROMPT_ID}")
    logger.info(f"Trials: {n_trials}, Temperature: {config['model']['temperature']}")
    logger.info(f"Total items: {len(data)}")
    logger.info(f"Expected API calls: {len(data) * n_trials}")

    # Execute
    if args.batch_collect:
        # Collect from a single previously submitted batch
        batch_id = args.batch_id
        if not batch_id:
            ids_path = f"{batch_output_dir}/{EXPERIMENT_TAG}_batch_id.json"
            if Path(ids_path).exists():
                with open(ids_path) as f:
                    batch_id = json.load(f)["batch_id"]
                logger.info(f"Loaded batch ID from {ids_path}: {batch_id}")
            else:
                logger.error(
                    "No batch ID. Use --batch-id or run --batch-submit-only first."
                )
                sys.exit(1)

        run_batch_collect(
            client, data, config, batch_id, batch_output_dir, output_dir
        )

    elif args.use_batch:
        if args.batch_submit_only:
            # Submit single batch (no chunking)
            run_batch_submit(client, data, config, batch_output_dir)
            logger.info(
                "Batch submitted. Run with --batch-collect to retrieve results."
            )
        else:
            # Chunked execution (default)
            chunk_size = (
                args.chunk_size
                or config["api"].get("batch_chunk_size", 5000)
            )
            run_batch_chunked(
                client, data, config,
                batch_output_dir, output_dir,
                chunk_size=chunk_size,
            )

    else:
        run_sync(client, data, config, output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

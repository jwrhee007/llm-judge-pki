#!/usr/bin/env python3
"""
Step 6: Exp. 2-3 Prompt Mitigation PKI 측정.

P-Lee-Standard 대비 P-Lee-Direct, P-Lee-CoT 프롬프트의 PKI 억제 효과를 비교한다.
Exp. 2-2 (05_context_swap.py)의 Same-Type swap 조건을 그대로 사용하되,
judge 프롬프트만 교체. Cross-Type은 H-PKI-2 기각으로 사용하지 않는다.

Verdict 해석은 Exp. 2-2와 동일:
  CORRECT       → PKI 발동 (parametric knowledge 누설)
  NOT_ATTEMPTED → Context-faithful (정상)
  INCORRECT    → 과잉 기각 (over-rejection)

Standard는 본 스크립트에서 돌리지 않는다 — Exp. 2-2 Same-Type 결과
(results/context_swap/same_swap_*) 를 baseline으로 재사용한다.

Usage:
    # P-Lee-Direct
    python scripts/06_prompt_mitigation.py --prompt-id P-Lee-Direct --use-batch

    # P-Lee-CoT (max_tokens 512 자동 적용)
    python scripts/06_prompt_mitigation.py --prompt-id P-Lee-CoT --use-batch --chunk-size 2000

    # Smoke test (10 items)
    python scripts/06_prompt_mitigation.py --prompt-id P-Lee-Direct --smoke-test
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
from src.data.context_swap import build_swap_pairs
from src.data.sampler import load_sampled_data, save_sampled_data
from src.evaluation.judge_runner import (
    classify_judge_batch,
    compute_judge_summary,
    parse_judge_batch_results,
    prepare_judge_batch_requests,
    run_judge_all,
    save_judge_results,
)
from src.utils.logger import setup_logger

logger = setup_logger("prompt_mitigation")

SWAP_TYPE = "same"  # H-PKI-2 기각으로 Same만 사용
SUPPORTED_PROMPTS = ["P-Lee-Direct", "P-Lee-CoT"]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_swap_data(
    data: list[dict],
    swap_type: str,
    seed: int = 42,
) -> list[dict]:
    """
    Swap pairs를 생성하고, Judge 평가용 데이터를 구성한다.

    context를 context_swap으로 교체한 새 리스트를 반환한다.
    (원본 데이터는 변경하지 않음)
    """
    import copy

    # Build swap pairs (원본에 swap 정보 추가)
    data = build_swap_pairs(
        data=data,
        swap_type=swap_type,
        seed=seed,
    )

    # Valid swap만 추출하고, context를 swap context로 교체
    swap_data = []
    for item in data:
        if not item.get("swap_valid", False):
            continue
        swapped = copy.deepcopy(item)
        swapped["context_original"] = swapped["context"]
        swapped["context"] = swapped["context_swap"]
        swap_data.append(swapped)

    logger.info(f"Swap data prepared: {len(swap_data)} valid items (swap_type={swap_type})")
    return swap_data


def log_pki_summary(
    summary: dict,
    swap_type: str,
    prompt_id: str,
    baseline_summary: dict | None = None,
) -> None:
    """Print PKI analysis summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Exp. 2-3 Prompt Mitigation Summary ({prompt_id}, {swap_type.upper()})")
    logger.info(f"{'='*60}")
    logger.info(f"Total items: {summary['total']}")

    # ACC_swap (NOT_ATTEMPTED = context-faithful = CAR)
    na_count = summary["majority_distribution"].get("NOT_ATTEMPTED", 0)
    acc_swap = na_count / summary["total"] if summary["total"] > 0 else 0
    logger.info(f"CAR (majority=NOT_ATTEMPTED): {acc_swap:.4f} ({na_count}/{summary['total']})")

    # PKI Rate (CORRECT under swapped context)
    correct_count = summary["majority_distribution"].get("CORRECT", 0)
    pki_rate = correct_count / summary["total"] if summary["total"] > 0 else 0
    logger.info(f"PKI Rate (majority=CORRECT): {pki_rate:.4f} ({correct_count}/{summary['total']})")

    # ORR (INCORRECT)
    incorrect_count = summary["majority_distribution"].get("INCORRECT", 0)
    orr = incorrect_count / summary["total"] if summary["total"] > 0 else 0
    logger.info(f"ORR (majority=INCORRECT): {orr:.4f} ({incorrect_count}/{summary['total']})")

    # Baseline (Exp.2-2 Same-Type) PKI 비교
    if baseline_summary:
        base_pki = baseline_summary["majority_distribution"].get("CORRECT", 0) / baseline_summary["total"]
        delta_pki = pki_rate - base_pki
        logger.info(
            f"\nBaseline (Exp.2-2 Same-Type, P-Lee-Standard) PKI: {base_pki:.4f}"
        )
        logger.info(
            f"ΔPKI = {prompt_id} - Standard = {pki_rate:.4f} - {base_pki:.4f} = {delta_pki:+.4f}"
        )

    # Entropy
    logger.info(f"\nMean verdict entropy: {summary['mean_entropy']:.4f}")
    logger.info(f"Unstable items (H>0): {summary['unstable_count']} ({summary['unstable_ratio']:.1%})")

    # By NER tag
    logger.info(f"\nBy NER/analysis tag:")
    for tag, stats in sorted(summary["by_tag"].items(), key=lambda x: -x[1]["total"]):
        logger.info(
            f"  {tag:15s}: n={stats['total']:4d}, "
            f"CAR={stats['accuracy']:.3f}, "
            f"H_mean={stats['mean_entropy']:.4f}, "
            f"unstable={stats['unstable_count']}"
        )


# =====================================================================
# Batch execution (chunked, with resume)
# =====================================================================

def run_swap_experiment(
    client: OpenAIClient,
    swap_data: list[dict],
    config: dict,
    swap_type: str,
    prompt_id: str,
    max_tokens: int,
    batch_output_dir: str,
    output_dir: str,
    chunk_size: int = 3000,
    use_batch: bool = True,
) -> list[dict]:
    """Run the prompt-mitigation judge evaluation (Same-Type swap)."""
    model = config["model"]["judge"]
    n_trials = config.get("judge", {}).get("n_trials", 30)
    temperature = config["model"]["temperature"]
    seed = config["model"]["seed"]
    experiment_tag = f"exp23_{prompt_id}_{swap_type}_swap"

    logger.info(
        f"\n--- Running {prompt_id} on {swap_type.upper()} Swap ---\n"
        f"  Model: {model}, Prompt: {prompt_id}\n"
        f"  Trials: {n_trials}, T={temperature}, max_tokens={max_tokens}\n"
        f"  Items: {len(swap_data)}"
    )

    if not use_batch:
        # Synchronous
        results = run_judge_all(
            client=client,
            data=swap_data,
            prompt_id=prompt_id,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
    else:
        # Chunked batch
        all_requests = prepare_judge_batch_requests(
            data=swap_data,
            prompt_id=prompt_id,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            experiment_tag=experiment_tag,
        )

        total = len(all_requests)
        n_chunks = (total + chunk_size - 1) // chunk_size

        logger.info(
            f"  Total requests: {total}, chunk_size: {chunk_size}, "
            f"n_chunks: {n_chunks}"
        )

        # Load progress for resume
        progress = _load_progress(batch_output_dir, experiment_tag)
        completed_chunks: dict[int, str] = {}
        if progress:
            for entry in progress.get("chunks", []):
                if entry.get("status") == "completed":
                    completed_chunks[entry["chunk_idx"]] = entry["batch_id"]
            if completed_chunks:
                logger.info(
                    f"  Resuming: {len(completed_chunks)}/{n_chunks} completed"
                )

        all_raw_results = []
        chunk_records = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)

            if chunk_idx in completed_chunks:
                logger.info(f"  Chunk {chunk_idx+1}/{n_chunks}: cached")
                out_path = (
                    f"{batch_output_dir}/"
                    f"{experiment_tag}_chunk{chunk_idx}_output.jsonl"
                )
                if Path(out_path).exists():
                    with open(out_path, "r") as f:
                        chunk_results = [json.loads(l) for l in f]
                else:
                    chunk_results = client.download_batch_results(
                        batch_id=completed_chunks[chunk_idx],
                        output_path=out_path,
                    )
                all_raw_results.extend(chunk_results)
                chunk_records.append({
                    "chunk_idx": chunk_idx,
                    "batch_id": completed_chunks[chunk_idx],
                    "status": "completed",
                })
                continue

            chunk_requests = all_requests[start:end]
            logger.info(
                f"  Chunk {chunk_idx+1}/{n_chunks} "
                f"({len(chunk_requests)} requests)"
            )

            chunk_file = client.create_batch_file(
                requests=chunk_requests,
                output_path=(
                    f"{batch_output_dir}/"
                    f"{experiment_tag}_chunk{chunk_idx}_input.jsonl"
                ),
            )
            batch_id = client.submit_batch(
                chunk_file,
                description=f"{experiment_tag} chunk {chunk_idx+1}/{n_chunks}",
            )
            logger.info(f"  Submitted: {batch_id}")

            batch = client.poll_batch(batch_id, poll_interval=30)
            if batch.status != "completed":
                logger.error(f"  Chunk {chunk_idx+1} failed: {batch.status}")
                chunk_records.append({
                    "chunk_idx": chunk_idx,
                    "batch_id": batch_id,
                    "status": batch.status,
                })
                _save_progress(chunk_records, n_chunks, batch_output_dir, experiment_tag)
                return []

            chunk_results = client.download_batch_results(
                batch_id=batch_id,
                output_path=(
                    f"{batch_output_dir}/"
                    f"{experiment_tag}_chunk{chunk_idx}_output.jsonl"
                ),
            )
            all_raw_results.extend(chunk_results)

            chunk_records.append({
                "chunk_idx": chunk_idx,
                "batch_id": batch_id,
                "status": "completed",
            })
            _save_progress(chunk_records, n_chunks, batch_output_dir, experiment_tag)

            logger.info(
                f"  Chunk {chunk_idx+1}/{n_chunks} done. "
                f"Cumulative: {len(all_raw_results)}"
            )

        # Parse
        grouped = parse_judge_batch_results(
            batch_results=all_raw_results,
            experiment_tag=experiment_tag,
        )
        results = classify_judge_batch(
            grouped_results=grouped,
            data=swap_data,
            prompt_id=prompt_id,
            n_trials=n_trials,
        )

    # Save results
    result_path = f"{output_dir}/{swap_type}_swap_results.jsonl"
    save_judge_results(results, result_path)

    # Summary — ground truth is NOT_ATTEMPTED for swapped context
    summary = compute_judge_summary(results, ground_truth="NOT_ATTEMPTED")

    summary_path = f"{output_dir}/{swap_type}_swap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    # Load Exp.2-2 Same-Type baseline (P-Lee-Standard) for ΔPKI
    baseline_path = "results/context_swap/same_swap_summary.json"
    baseline_summary = None
    if Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline_summary = json.load(f)
    else:
        logger.warning(
            f"Baseline (Exp.2-2 Same-Type) not found at {baseline_path}; "
            f"ΔPKI will be skipped."
        )

    log_pki_summary(summary, swap_type, prompt_id, baseline_summary)

    return results


def _save_progress(records, n_total, batch_dir, tag):
    path = f"{batch_dir}/{tag}_chunks.json"
    with open(path, "w") as f:
        json.dump({
            "n_chunks_total": n_total,
            "n_completed": sum(1 for c in records if c["status"] == "completed"),
            "chunks": records,
        }, f, indent=2)


def _load_progress(batch_dir, tag):
    path = f"{batch_dir}/{tag}_chunks.json"
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp. 2-3: Prompt Mitigation PKI Measurement (Same-Type swap only)",
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        choices=SUPPORTED_PROMPTS,
        help=(
            "Judge prompt to evaluate. P-Lee-Standard is excluded — "
            "reuse Exp.2-2 Same-Type result (results/context_swap/same_swap_*) instead."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_sampled.jsonl",
        help="Input sampled data file (must match Exp.2-2)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to results/prompt_mitigation/{prompt_id}/."
        ),
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use OpenAI Batch API (chunked)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Batch chunk size (default: from config)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with only 10 items",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    # max_tokens 분기 — silent default 금지, 키 없으면 KeyError
    if args.prompt_id == "P-Lee-CoT":
        if "max_tokens_judge_cot" not in config["model"]:
            raise KeyError(
                "config.model.max_tokens_judge_cot is required for P-Lee-CoT"
            )
        max_tokens = config["model"]["max_tokens_judge_cot"]
    elif args.prompt_id == "P-Lee-Direct":
        if "max_tokens_judge" not in config["model"]:
            raise KeyError(
                "config.model.max_tokens_judge is required for P-Lee-Direct"
            )
        max_tokens = config["model"]["max_tokens_judge"]
    else:
        # Defensive: argparse choices should prevent this branch
        raise ValueError(f"Unsupported prompt_id: {args.prompt_id}")

    # Output directory: auto-compose if not provided
    output_dir = args.output_dir or f"results/prompt_mitigation/{args.prompt_id}"
    batch_output_dir = config["api"]["batch_output_dir"]
    chunk_size = args.chunk_size or config["api"].get("batch_chunk_size", 5000)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(batch_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Exp. 2-3: Prompt Mitigation PKI Measurement")
    logger.info(f"  prompt_id   : {args.prompt_id}")
    logger.info(f"  swap_type   : {SWAP_TYPE} (hardcoded — H-PKI-2 기각)")
    logger.info(f"  max_tokens  : {max_tokens}")
    logger.info(f"  output_dir  : {output_dir}")
    logger.info("=" * 60)

    # Load data
    data = load_sampled_data(args.input)

    if args.smoke_test:
        data = data[:10]
        logger.info(f"Smoke test mode: {len(data)} items")

    # Initialize client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    client = OpenAIClient(
        api_key=api_key,
        max_retries=config["api"]["max_retries"],
        requests_per_minute=config["api"]["requests_per_minute"],
    )

    # Prepare swapped data (Same-Type only)
    swap_data = prepare_swap_data(
        data, swap_type=SWAP_TYPE, seed=config["model"]["seed"]
    )

    # Save swap metadata (same filename pattern as 05 for tooling compat)
    swap_meta_path = f"{output_dir}/{SWAP_TYPE}_swap_data.jsonl"
    save_sampled_data(swap_data, swap_meta_path)

    # Run experiment
    run_swap_experiment(
        client=client,
        swap_data=swap_data,
        config=config,
        swap_type=SWAP_TYPE,
        prompt_id=args.prompt_id,
        max_tokens=max_tokens,
        batch_output_dir=batch_output_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        use_batch=args.use_batch,
    )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

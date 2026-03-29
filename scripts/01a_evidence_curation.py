#!/usr/bin/env python3
"""
Step 1a: Evidence Curation — Judge-as-Verifier.

String-based evidence_present를 통과한 TriviaQA 문항에 대해,
실험에 사용할 동일한 Judge와 동일한 프롬프트(P-Lee-Standard, T=0)로
1회 평가하여 CORRECT 판정을 받은 문항만 선별한다.

이를 통해 "Curated Evidence-Centric Subset"을 구성한다.

파이프라인:
  00_prepare_data.py  → evidence_present 필터 (Stage 1)
  01a_evidence_curation.py → Judge-verified grounding (Stage 2)  ← 이 스크립트
  01_ner_tagging.py   → NER 태깅 + 랜덤 샘플링 (Stage 3)

Usage:
    # Batch API — chunked (권장)
    python scripts/01a_evidence_curation.py --use-batch

    # chunk 크기 조절
    python scripts/01a_evidence_curation.py --use-batch --chunk-size 500

    # 동기 방식
    python scripts/01a_evidence_curation.py

    # 스모크 테스트
    python scripts/01a_evidence_curation.py --smoke-test
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.openai_client import OpenAIClient
from src.prompts.judge_prompts import build_judge_messages, parse_verdict
from src.utils.logger import setup_logger

logger = setup_logger("evidence_curation")

EXPERIMENT_TAG = "evidence_curation"
PROMPT_ID = "P-Lee-Standard"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(input_path: str) -> list[dict]:
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} items from {input_path}")
    return data


def save_data(data: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} items to {output}")


# =====================================================================
# Synchronous execution
# =====================================================================

def run_sync(
    client: OpenAIClient,
    data: list[dict],
    model: str,
    max_tokens: int = 16,
) -> list[dict]:
    """Synchronously verify each item with 1 Judge call."""
    for item in tqdm(data, desc="Evidence curation"):
        messages = build_judge_messages(
            question=item["question"],
            context=item["context"],
            candidate_answer=item["answer_value"],
            prompt_id=PROMPT_ID,
        )
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            verdict = parse_verdict(response, PROMPT_ID)
            item["curation_verdict"] = verdict
            item["curation_response"] = response.strip()
        except Exception as e:
            logger.warning(f"Failed for {item['question_id']}: {e}")
            item["curation_verdict"] = "ERROR"
            item["curation_response"] = str(e)

    return data


# =====================================================================
# Batch API execution (chunked, with resume)
# =====================================================================

def prepare_batch_requests(
    data: list[dict],
    model: str,
    max_tokens: int = 16,
) -> list[dict]:
    """Prepare batch requests — 1 request per item."""
    requests = []
    for data_idx, item in enumerate(data):
        messages = build_judge_messages(
            question=item["question"],
            context=item["context"],
            candidate_answer=item["answer_value"],
            prompt_id=PROMPT_ID,
        )
        requests.append({
            "custom_id": f"{EXPERIMENT_TAG}_idx{data_idx}",
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        })
    logger.info(f"Prepared {len(requests)} curation requests")
    return requests


def run_batch_chunked(
    client: OpenAIClient,
    data: list[dict],
    model: str,
    batch_output_dir: str,
    chunk_size: int = 500,
    max_tokens: int = 16,
) -> list[dict]:
    """Run curation via chunked Batch API with resume support."""
    all_requests = prepare_batch_requests(data, model, max_tokens)
    total = len(all_requests)
    n_chunks = (total + chunk_size - 1) // chunk_size

    logger.info(
        f"Chunked curation: {total} requests, "
        f"chunk_size={chunk_size}, n_chunks={n_chunks}"
    )

    # Load progress for resume
    progress = _load_progress(batch_output_dir)
    completed_chunks: dict[int, str] = {}
    if progress:
        for entry in progress.get("chunks", []):
            if entry.get("status") == "completed":
                completed_chunks[entry["chunk_idx"]] = entry["batch_id"]
        if completed_chunks:
            logger.info(
                f"Resuming: {len(completed_chunks)}/{n_chunks} chunks completed"
            )

    all_raw_results = []
    chunk_records = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)

        # Already completed → load cached
        if chunk_idx in completed_chunks:
            logger.info(
                f"Chunk {chunk_idx + 1}/{n_chunks}: already completed, loading"
            )
            output_path = (
                f"{batch_output_dir}/"
                f"{EXPERIMENT_TAG}_chunk{chunk_idx}_output.jsonl"
            )
            if Path(output_path).exists():
                with open(output_path, "r") as f:
                    chunk_results = [json.loads(line) for line in f]
            else:
                chunk_results = client.download_batch_results(
                    batch_id=completed_chunks[chunk_idx],
                    output_path=output_path,
                )
            all_raw_results.extend(chunk_results)
            chunk_records.append({
                "chunk_idx": chunk_idx,
                "batch_id": completed_chunks[chunk_idx],
                "status": "completed",
            })
            continue

        # New chunk
        chunk_requests = all_requests[start:end]
        logger.info(
            f"Chunk {chunk_idx + 1}/{n_chunks} "
            f"(requests {start}~{end - 1}, n={len(chunk_requests)})"
        )

        # Submit
        chunk_file = client.create_batch_file(
            requests=chunk_requests,
            output_path=(
                f"{batch_output_dir}/"
                f"{EXPERIMENT_TAG}_chunk{chunk_idx}_input.jsonl"
            ),
        )
        batch_id = client.submit_batch(
            chunk_file,
            description=(
                f"{EXPERIMENT_TAG} chunk {chunk_idx + 1}/{n_chunks}"
            ),
        )
        logger.info(f"Submitted: {batch_id}")

        # Poll
        batch = client.poll_batch(batch_id, poll_interval=15)
        if batch.status != "completed":
            logger.error(
                f"Chunk {chunk_idx + 1} failed: {batch.status}. "
                f"Re-run to resume."
            )
            chunk_records.append({
                "chunk_idx": chunk_idx,
                "batch_id": batch_id,
                "status": batch.status,
            })
            _save_progress(chunk_records, n_chunks, batch_output_dir)
            return data

        # Download
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
        })
        _save_progress(chunk_records, n_chunks, batch_output_dir)

        logger.info(
            f"Chunk {chunk_idx + 1}/{n_chunks} complete. "
            f"Cumulative: {len(all_raw_results)}"
        )

    # Parse results and merge into data
    pattern = re.compile(r"^evidence_curation_idx(\d+)$")
    for result in all_raw_results:
        m = pattern.match(result["custom_id"])
        if not m:
            continue
        data_idx = int(m.group(1))
        if data_idx >= len(data):
            continue

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
            verdict = parse_verdict(response_text, PROMPT_ID)
            data[data_idx]["curation_verdict"] = verdict
            data[data_idx]["curation_response"] = response_text.strip()
        else:
            data[data_idx]["curation_verdict"] = "ERROR"
            data[data_idx]["curation_response"] = "NO_RESPONSE"

    return data


def _save_progress(
    chunk_records: list[dict],
    n_chunks_total: int,
    batch_output_dir: str,
) -> None:
    path = f"{batch_output_dir}/{EXPERIMENT_TAG}_chunks.json"
    with open(path, "w") as f:
        json.dump({
            "n_chunks_total": n_chunks_total,
            "n_completed": sum(
                1 for c in chunk_records if c["status"] == "completed"
            ),
            "chunks": chunk_records,
        }, f, indent=2)


def _load_progress(batch_output_dir: str) -> dict | None:
    path = f"{batch_output_dir}/{EXPERIMENT_TAG}_chunks.json"
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Evidence curation (Judge-as-Verifier)",
    )
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_evidence_present.jsonl",
        help="Input: Stage 1 output (string-based evidence_present)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/triviaqa_rc_curated.jsonl",
        help="Output: curated subset (CORRECT only)",
    )
    parser.add_argument(
        "--output-full",
        default="data/processed/triviaqa_rc_curated_full.jsonl",
        help="Output: all items with curation_verdict field",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use OpenAI Batch API (chunked)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Batch chunk size (default: 500)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with only 20 items",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    batch_output_dir = config["api"]["batch_output_dir"]
    Path(batch_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stage 2: Evidence Curation (Judge-as-Verifier)")
    logger.info("=" * 60)

    # Load Stage 1 output
    data = load_data(args.input)

    if args.smoke_test:
        data = data[:20]
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

    model = config["model"]["judge"]
    max_tokens = config["model"].get("max_tokens_judge", 16)
    logger.info(f"Judge model: {model}")
    logger.info(f"Prompt: {PROMPT_ID}, T=0, 1 call per item")
    logger.info(f"Total items: {len(data)}")

    # Execute
    if args.use_batch:
        data = run_batch_chunked(
            client=client,
            data=data,
            model=model,
            batch_output_dir=batch_output_dir,
            chunk_size=args.chunk_size,
            max_tokens=max_tokens,
        )
    else:
        data = run_sync(
            client=client,
            data=data,
            model=model,
            max_tokens=max_tokens,
        )

    # Summary
    from collections import Counter
    verdict_dist = Counter(d.get("curation_verdict", "MISSING") for d in data)

    correct_count = verdict_dist.get("CORRECT", 0)
    incorrect_count = verdict_dist.get("INCORRECT", 0)
    not_attempted_count = verdict_dist.get("NOT_ATTEMPTED", 0)
    error_count = verdict_dist.get("ERROR", 0) + verdict_dist.get("MISSING", 0)

    logger.info(f"\n{'='*50}")
    logger.info("Evidence Curation Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total items:         {len(data)}")
    logger.info(f"CORRECT (curated):   {correct_count} ({correct_count/len(data)*100:.1f}%)")
    logger.info(f"INCORRECT:           {incorrect_count} ({incorrect_count/len(data)*100:.1f}%)")
    logger.info(f"NOT_ATTEMPTED:       {not_attempted_count} ({not_attempted_count/len(data)*100:.1f}%)")
    if error_count:
        logger.info(f"ERRORS:              {error_count}")

    # Save full data (with curation_verdict)
    save_data(data, args.output_full)

    # Save curated subset (CORRECT only)
    curated = [d for d in data if d.get("curation_verdict") == "CORRECT"]
    save_data(curated, args.output)

    logger.info(
        f"\nCurated subset: {len(curated)} items saved to {args.output}"
    )
    logger.info(
        f"Curation rate: {len(curated)/len(data)*100:.1f}% "
        f"({len(curated)}/{len(data)})"
    )

    # Next steps
    logger.info(
        f"\n다음 단계:\n"
        f"  python scripts/01_ner_tagging.py "
        f"--input {args.output} --skip-ner\n"
        f"  (NER 태깅이 필요하면 --skip-ner 제거)"
    )


if __name__ == "__main__":
    main()

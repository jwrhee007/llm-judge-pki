"""
Judge Evaluation Runner.

(question, context, candidate_answer) 트리플렛에 대해 Judge 모델의
verdict를 생성하고, 30회 반복으로 verdict 분포를 측정한다.

지원 모드:
  - sync: 동기 실행 (디버깅/스모크 테스트)
  - batch: OpenAI Batch API (대규모 실행)
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.api.openai_client import OpenAIClient
from src.prompts.judge_prompts import build_judge_messages, parse_verdict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

VALID_VERDICTS = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}


# -----------------------------------------------------------------------
# Verdict analysis utilities
# -----------------------------------------------------------------------

def compute_verdict_entropy(verdicts: list[str]) -> float:
    """
    Verdict 분포의 Shannon entropy를 산출한다.

    H = -Σ p_i * ln(p_i)  (p_i > 0)

    H=0이면 결정적(모두 동일), H>0이면 확률적 경쟁.
    """
    n = len(verdicts)
    if n == 0:
        return 0.0
    counts = Counter(verdicts)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * np.log(p)
    return entropy


def compute_majority_verdict(verdicts: list[str]) -> str:
    """30회 반복 중 과반수 verdict를 반환한다."""
    counts = Counter(verdicts)
    return counts.most_common(1)[0][0]


def compute_verdict_distribution(verdicts: list[str]) -> dict[str, int]:
    """Verdict 분포를 dict로 반환한다."""
    dist = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0, "PARSE_ERROR": 0}
    for v in verdicts:
        dist[v] = dist.get(v, 0) + 1
    return dist


# -----------------------------------------------------------------------
# Single item evaluation (sync)
# -----------------------------------------------------------------------

def evaluate_single(
    client: OpenAIClient,
    item: dict,
    prompt_id: str,
    model: str,
    n_trials: int = 30,
    temperature: float = 0,
    max_tokens: int = 16,
    seed: int = 42,
) -> dict:
    """
    단일 문항에 대해 n_trials회 Judge 평가를 수행한다.

    Args:
        item: dict with question, context, answer_value
        prompt_id: "P-Lee-Standard" | "P-Lee-Direct" | "P-Lee-CoT"
        n_trials: 반복 횟수
        temperature: 0 for judge experiments
        max_tokens: 16 for Standard/Direct, 512 for CoT
    """
    messages = build_judge_messages(
        question=item["question"],
        context=item["context"],
        candidate_answer=item["answer_value"],
        prompt_id=prompt_id,
    )

    trials = []
    for trial_idx in range(n_trials):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            verdict = parse_verdict(response, prompt_id)
            trials.append({
                "trial_idx": trial_idx,
                "response": response,
                "verdict": verdict,
            })
        except Exception as e:
            logger.error(
                f"Judge eval failed: {item['question_id']} "
                f"(trial={trial_idx}): {e}"
            )
            trials.append({
                "trial_idx": trial_idx,
                "response": None,
                "verdict": "PARSE_ERROR",
            })

    verdicts = [t["verdict"] for t in trials]

    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "answer_value": item["answer_value"],
        "ner_tag": item.get("ner_tag", "UNKNOWN"),
        "analysis_tag": item.get("analysis_tag", "UNKNOWN"),
        "prompt_id": prompt_id,
        "n_trials": n_trials,
        "majority_verdict": compute_majority_verdict(verdicts),
        "verdict_entropy": compute_verdict_entropy(verdicts),
        "verdict_distribution": compute_verdict_distribution(verdicts),
        "trials": trials,
    }


# -----------------------------------------------------------------------
# Batch API: prepare requests
# -----------------------------------------------------------------------

def prepare_judge_batch_requests(
    data: list[dict],
    prompt_id: str,
    model: str,
    n_trials: int = 30,
    temperature: float = 0,
    max_tokens: int = 16,
    seed: int = 42,
    experiment_tag: str = "baseline",
) -> list[dict]:
    """
    Batch API용 Judge 평가 요청 리스트를 생성한다.

    custom_id: {experiment_tag}_{prompt_id}_idx{data_idx}_t{trial_idx}
    """
    requests = []
    for data_idx, item in enumerate(data):
        messages = build_judge_messages(
            question=item["question"],
            context=item["context"],
            candidate_answer=item["answer_value"],
            prompt_id=prompt_id,
        )

        for trial_idx in range(n_trials):
            custom_id = (
                f"{experiment_tag}_{prompt_id}_idx{data_idx}_t{trial_idx}"
            )
            req = {
                "custom_id": custom_id,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                req["seed"] = seed
            requests.append(req)

    logger.info(
        f"Prepared {len(requests)} batch requests "
        f"({experiment_tag}, {prompt_id}, {len(data)} items × {n_trials} trials)"
    )
    return requests


# -----------------------------------------------------------------------
# Batch API: parse results
# -----------------------------------------------------------------------

def parse_judge_batch_results(
    batch_results: list[dict],
    experiment_tag: str = "baseline",
) -> dict[int, list]:
    """
    Batch API 결과를 data_index별로 그룹핑한다.

    Returns:
        dict: data_index(int) → list of (trial_idx, response_text)
    """
    pattern = re.compile(
        rf"^{re.escape(experiment_tag)}_[^_]+(?:_[^_]+)*_idx(\d+)_t(\d+)$"
    )

    grouped: dict[int, list] = {}
    parse_failures = 0

    for result in batch_results:
        custom_id = result["custom_id"]
        m = pattern.match(custom_id)
        if not m:
            # Try a more flexible pattern
            parts = custom_id.rsplit("_idx", 1)
            if len(parts) == 2:
                idx_trial = parts[1].split("_t")
                if len(idx_trial) == 2:
                    data_idx = int(idx_trial[0])
                    trial_idx = int(idx_trial[1])
                else:
                    parse_failures += 1
                    continue
            else:
                parse_failures += 1
                continue
        else:
            data_idx = int(m.group(1))
            trial_idx = int(m.group(2))

        # Extract response text
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
        else:
            response_text = None

        if data_idx not in grouped:
            grouped[data_idx] = []
        grouped[data_idx].append((trial_idx, response_text))

    if parse_failures > 0:
        logger.warning(f"Failed to parse {parse_failures} custom_ids")

    logger.info(f"Parsed {len(grouped)} item groups from batch results")
    return grouped


def classify_judge_batch(
    grouped_results: dict[int, list],
    data: list[dict],
    prompt_id: str,
    n_trials: int = 30,
) -> list[dict]:
    """
    Batch 결과로부터 verdict 파싱 + 통계 산출을 수행한다.
    """
    results = []

    for data_idx, trials_raw in tqdm(
        sorted(grouped_results.items()),
        desc=f"Parsing verdicts ({prompt_id})",
    ):
        if data_idx >= len(data):
            logger.warning(f"Data index out of range: {data_idx}")
            continue

        item = data[data_idx]
        trials = []

        for trial_idx, response_text in sorted(trials_raw, key=lambda x: x[0]):
            if response_text is None:
                verdict = "PARSE_ERROR"
            else:
                verdict = parse_verdict(response_text, prompt_id)

            trials.append({
                "trial_idx": trial_idx,
                "response": response_text,
                "verdict": verdict,
            })

        verdicts = [t["verdict"] for t in trials]

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answer_value": item["answer_value"],
            "ner_tag": item.get("ner_tag", "UNKNOWN"),
            "analysis_tag": item.get("analysis_tag", "UNKNOWN"),
            "prompt_id": prompt_id,
            "n_trials": n_trials,
            "majority_verdict": compute_majority_verdict(verdicts),
            "verdict_entropy": compute_verdict_entropy(verdicts),
            "verdict_distribution": compute_verdict_distribution(verdicts),
            "trials": trials,
        })

    return results


# -----------------------------------------------------------------------
# Full sync run
# -----------------------------------------------------------------------

def run_judge_all(
    client: OpenAIClient,
    data: list[dict],
    prompt_id: str,
    model: str,
    n_trials: int = 30,
    temperature: float = 0,
    max_tokens: int = 16,
    seed: int = 42,
) -> list[dict]:
    """전체 데이터셋에 대해 동기 Judge 평가를 수행한다."""
    results = []
    for item in tqdm(data, desc=f"Judge eval ({prompt_id})"):
        result = evaluate_single(
            client=client,
            item=item,
            prompt_id=prompt_id,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        results.append(result)
    return results


# -----------------------------------------------------------------------
# Result I/O
# -----------------------------------------------------------------------

def save_judge_results(results: list[dict], output_path: str) -> None:
    """Save judge results to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} judge results to {output}")


def load_judge_results(input_path: str) -> list[dict]:
    """Load judge results from JSONL."""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} judge results from {input_path}")
    return results


# -----------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------

def compute_judge_summary(results: list[dict], ground_truth: str = "CORRECT") -> dict:
    """
    Judge 결과의 요약 통계를 산출한다.

    Args:
        results: judge result dicts
        ground_truth: 기대 verdict (Baseline에서는 "CORRECT")

    Returns:
        dict with accuracy, verdict distribution, entropy stats
    """
    n = len(results)
    if n == 0:
        return {}

    # Majority verdict accuracy
    correct_count = sum(
        1 for r in results if r["majority_verdict"] == ground_truth
    )
    accuracy = correct_count / n

    # Overall verdict distribution (majority)
    majority_dist = Counter(r["majority_verdict"] for r in results)

    # Entropy statistics
    entropies = [r["verdict_entropy"] for r in results]
    unstable_count = sum(1 for e in entropies if e > 0)

    # By NER tag
    by_tag: dict[str, dict] = {}
    for r in results:
        tag = r.get("analysis_tag", r.get("ner_tag", "UNKNOWN"))
        if tag not in by_tag:
            by_tag[tag] = {"total": 0, "correct": 0, "entropies": []}
        by_tag[tag]["total"] += 1
        if r["majority_verdict"] == ground_truth:
            by_tag[tag]["correct"] += 1
        by_tag[tag]["entropies"].append(r["verdict_entropy"])

    tag_summary = {}
    for tag, stats in by_tag.items():
        tag_summary[tag] = {
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "mean_entropy": float(np.mean(stats["entropies"])),
            "unstable_count": sum(1 for e in stats["entropies"] if e > 0),
        }

    return {
        "total": n,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "majority_distribution": dict(majority_dist),
        "mean_entropy": float(np.mean(entropies)),
        "median_entropy": float(np.median(entropies)),
        "unstable_count": unstable_count,
        "unstable_ratio": unstable_count / n,
        "by_tag": tag_summary,
    }

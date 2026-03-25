"""
Experiment 2-0: Knowledge Probe.

Judge에게 context 없이 질문만 제시하여 사전지식 보유 여부를 확인한다.
Self-consistency 3회를 적용하여 hallucination과 실제 지식을 구분한다.

방식 A (Bare question) vs 방식 B (Knowledge-eliciting prompt)를 비교하고,
PKI rate와의 상관이 높은 방식을 채택한다.

4단계 분류:
    - strong-knows (3/3): 확실하게 인코딩된 지식
    - weak-knows  (2/3): 불확실하지만 접근 가능
    - guess       (1/3): 우연히 맞춘 가능성
    - doesn't-know(0/3): 해당 지식 미보유
"""

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.api.openai_client import OpenAIClient
from src.probes.answer_matcher import match_answer
from src.prompts.probe_prompts import (
    build_probe_messages_method_a,
    build_probe_messages_method_b,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Classification thresholds ---
CLASSIFICATION_MAP = {
    3: "strong-knows",
    2: "weak-knows",
    1: "guess",
    0: "doesn't-know",
}


def classify_knowledge(n_correct: int) -> str:
    """Classify knowledge level based on number of correct answers out of 3."""
    return CLASSIFICATION_MAP.get(n_correct, "doesn't-know")


# -----------------------------------------------------------------------
# Synchronous execution (per-item)
# -----------------------------------------------------------------------

def run_probe_single(
    client: OpenAIClient,
    item: dict,
    method: str,
    model: str,
    n_trials: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 256,
    match_model: str = "gpt-4o-mini-2024-07-18",
) -> dict:
    """
    단일 문항에 대해 Knowledge Probe를 실행한다.

    Args:
        client: OpenAI API 클라이언트
        item: 문항 데이터 (question, answer_value, answer_aliases 포함)
        method: "A" (bare question) 또는 "B" (knowledge-eliciting)
        model: Probe 대상 모델
        n_trials: Self-consistency 시행 횟수
        temperature: Sampling temperature
        max_tokens: 최대 토큰 수
        match_model: Answer matching에 사용할 모델

    Returns:
        dict with probe results
    """
    question = item["question"]
    gold_answer = item["answer_value"]
    aliases = item.get("answer_aliases", [])

    # Build messages
    if method == "A":
        messages = build_probe_messages_method_a(question)
    elif method == "B":
        messages = build_probe_messages_method_b(question)
    else:
        raise ValueError(f"Unknown probe method: {method}")

    # Run n_trials
    trial_results = []
    for trial_idx in range(n_trials):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=None,  # No seed for self-consistency sampling
            )

            match_result = match_answer(
                client=client,
                question=question,
                gold_answer=gold_answer,
                aliases=aliases,
                predicted=response,
                model=match_model,
                use_llm_fallback=True,
            )

            trial_results.append({
                "trial_idx": trial_idx,
                "response": response,
                "is_correct": match_result["is_correct"],
                "match_method": match_result["match_method"],
            })
        except Exception as e:
            logger.error(
                f"Probe failed for {item['question_id']} "
                f"(method={method}, trial={trial_idx}): {e}"
            )
            trial_results.append({
                "trial_idx": trial_idx,
                "response": None,
                "is_correct": False,
                "match_method": "error",
            })

    # Classify
    n_correct = sum(1 for t in trial_results if t["is_correct"])
    knowledge_level = classify_knowledge(n_correct)

    return {
        "question_id": item["question_id"],
        "question": question,
        "gold_answer": gold_answer,
        "method": method,
        "n_correct": n_correct,
        "n_trials": n_trials,
        "knowledge_level": knowledge_level,
        "trials": trial_results,
    }


# -----------------------------------------------------------------------
# Batch API execution
# -----------------------------------------------------------------------

def prepare_probe_batch_requests(
    data: list[dict],
    method: str,
    model: str,
    n_trials: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 256,
) -> tuple[list[dict], dict[int, str]]:
    """
    Batch API용 요청 리스트를 생성한다.

    custom_id 형식: probe_{method}_idx{data_index}_t{trial_idx}
    (question_id에 중복이 있을 수 있으므로 data index를 사용)

    Returns:
        tuple of (requests list, idx_to_question_id mapping)
    """
    requests = []
    idx_to_question_id: dict[int, str] = {}

    for data_idx, item in enumerate(data):
        question = item["question"]
        idx_to_question_id[data_idx] = item["question_id"]

        if method == "A":
            messages = build_probe_messages_method_a(question)
        elif method == "B":
            messages = build_probe_messages_method_b(question)
        else:
            raise ValueError(f"Unknown probe method: {method}")

        for trial_idx in range(n_trials):
            custom_id = f"probe_{method}_idx{data_idx}_t{trial_idx}"
            requests.append({
                "custom_id": custom_id,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })

    logger.info(
        f"Prepared {len(requests)} batch requests "
        f"(method={method}, {len(data)} items × {n_trials} trials)"
    )
    return requests, idx_to_question_id


def parse_probe_batch_results(
    batch_results: list[dict],
    data: list[dict],
    method: str,
    n_trials: int = 3,
) -> dict[str, list]:
    """
    Batch API 결과를 파싱하여 data_index별로 그룹핑한다.

    Returns:
        dict: data_index(str) → list of (trial_idx, response_text)
    """
    import re

    # 결과 그룹핑 (data_index 기준)
    grouped: dict[str, list] = {}
    pattern = re.compile(r"^probe_[AB]_idx(\d+)_t(\d+)$")

    for result in batch_results:
        custom_id = result["custom_id"]
        m = pattern.match(custom_id)
        if not m:
            logger.warning(f"Unexpected custom_id format: {custom_id}")
            continue

        data_idx = m.group(1)   # str key for consistency
        trial_idx = int(m.group(2))

        # Extract response text
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
        else:
            response_text = None
            logger.warning(f"No response for {custom_id}")

        if data_idx not in grouped:
            grouped[data_idx] = []
        grouped[data_idx].append((trial_idx, response_text))

    logger.info(f"Parsed {len(grouped)} question groups from batch results")
    return grouped


def classify_from_batch(
    grouped_results: dict[str, list],
    data: list[dict],
    client: OpenAIClient,
    method: str,
    match_model: str = "gpt-4o-mini-2024-07-18",
    n_trials: int = 3,
) -> list[dict]:
    """
    Batch 결과로부터 answer matching + 4단계 분류를 수행한다.
    grouped_results의 key는 data_index (str)이다.
    """
    results = []

    for data_idx_str, trials in tqdm(
        grouped_results.items(),
        desc=f"Matching (method={method})",
    ):
        data_idx = int(data_idx_str)
        if data_idx >= len(data):
            logger.warning(f"Data index out of range: {data_idx}")
            continue

        item = data[data_idx]

        trial_results = []
        for trial_idx, response_text in sorted(trials, key=lambda x: x[0]):
            if response_text is None:
                trial_results.append({
                    "trial_idx": trial_idx,
                    "response": None,
                    "is_correct": False,
                    "match_method": "error",
                })
                continue

            match_result = match_answer(
                client=client,
                question=item["question"],
                gold_answer=item["answer_value"],
                aliases=item.get("answer_aliases", []),
                predicted=response_text,
                model=match_model,
                use_llm_fallback=True,
            )

            trial_results.append({
                "trial_idx": trial_idx,
                "response": response_text,
                "is_correct": match_result["is_correct"],
                "match_method": match_result["match_method"],
            })

        n_correct = sum(1 for t in trial_results if t["is_correct"])
        knowledge_level = classify_knowledge(n_correct)

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "gold_answer": item["answer_value"],
            "ner_tag": item.get("ner_tag", "UNKNOWN"),
            "method": method,
            "n_correct": n_correct,
            "n_trials": n_trials,
            "knowledge_level": knowledge_level,
            "trials": trial_results,
        })

    return results


# -----------------------------------------------------------------------
# Full synchronous run (for small datasets or testing)
# -----------------------------------------------------------------------

def run_probe_all(
    client: OpenAIClient,
    data: list[dict],
    method: str,
    model: str,
    n_trials: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 256,
    match_model: str = "gpt-4o-mini-2024-07-18",
) -> list[dict]:
    """
    전체 데이터셋에 대해 Knowledge Probe를 실행한다.

    Returns:
        list of probe result dicts
    """
    results = []
    for item in tqdm(data, desc=f"Knowledge Probe (method={method})"):
        result = run_probe_single(
            client=client,
            item=item,
            method=method,
            model=model,
            n_trials=n_trials,
            temperature=temperature,
            max_tokens=max_tokens,
            match_model=match_model,
        )
        result["ner_tag"] = item.get("ner_tag", "UNKNOWN")
        results.append(result)

    return results


# -----------------------------------------------------------------------
# Result I/O
# -----------------------------------------------------------------------

def save_probe_results(results: list[dict], output_path: str) -> None:
    """Save probe results to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} probe results to {output}")


def load_probe_results(input_path: str) -> list[dict]:
    """Load probe results from JSONL."""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} probe results from {input_path}")
    return results


# -----------------------------------------------------------------------
# Analysis utilities
# -----------------------------------------------------------------------

def compute_probe_summary(results: list[dict]) -> dict:
    """
    Probe 결과의 4단계 분류 요약 통계를 산출한다.

    Returns:
        dict with:
            - overall: Counter of knowledge levels
            - by_ner_tag: dict[ner_tag -> Counter]
            - total: int
    """
    overall = Counter()
    by_ner_tag: dict[str, Counter] = {}

    for item in results:
        level = item["knowledge_level"]
        tag = item.get("ner_tag", "UNKNOWN")

        overall[level] += 1

        if tag not in by_ner_tag:
            by_ner_tag[tag] = Counter()
        by_ner_tag[tag][level] += 1

    return {
        "overall": dict(overall),
        "by_ner_tag": {tag: dict(counts) for tag, counts in by_ner_tag.items()},
        "total": len(results),
    }

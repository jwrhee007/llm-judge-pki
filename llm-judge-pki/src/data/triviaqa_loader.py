"""
TriviaQA rc 데이터 로딩 및 evidence_present 필터링.

TriviaQA rc (reading comprehension) subset에서 answer alias가
context에 명시적으로 포함된 문항만 추출한다.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _check_evidence_present(
    context: str,
    answer_aliases: list[str],
) -> bool:
    """
    Answer alias 중 하나라도 context에 포함되어 있으면 True.
    대소문자 무시, 공백 정규화 적용.
    """
    norm_ctx = _normalize_text(context)
    for alias in answer_aliases:
        if _normalize_text(alias) in norm_ctx:
            return True
    return False


def _extract_best_context(search_results: list[dict]) -> str | None:
    """
    TriviaQA rc의 entity_pages / search_results에서
    answer를 포함하는 첫 번째 context를 추출한다.
    """
    for result in search_results:
        if result.get("search_context"):
            return result["search_context"]
    return None


def load_triviaqa_rc(
    split: str = "validation",
    raw_dir: str = "data/raw",
    max_context_length: int = 3000,
) -> list[dict]:
    """
    TriviaQA rc subset을 로드하고 evidence_present 필터링을 수행한다.

    Returns:
        list of dict, each containing:
            - question_id: str
            - question: str
            - answer_value: str (normalized answer)
            - answer_aliases: list[str]
            - context: str (evidence가 포함된 context)
            - evidence_present: bool
    """
    logger.info(f"Loading TriviaQA rc ({split})...")
    ds = load_dataset("trivia_qa", "rc", split=split, trust_remote_code=True)
    logger.info(f"Raw dataset size: {len(ds)}")

    processed = []
    skipped_no_context = 0
    skipped_no_evidence = 0

    for idx, item in enumerate(ds):
        question = item["question"]
        answer = item["answer"]

        # answer_value와 aliases 추출
        answer_value = answer.get("value", "")
        answer_aliases = answer.get("aliases", [])
        if answer_value and answer_value not in answer_aliases:
            answer_aliases = [answer_value] + answer_aliases
        normalized_aliases = answer.get("normalized_aliases", [])
        all_aliases = list(set(answer_aliases + normalized_aliases))

        if not answer_value:
            continue

        # Context 추출: entity_pages와 search_results에서 탐색
        context = None

        # entity_pages (Wikipedia) 우선 탐색
        entity_pages = item.get("entity_pages", {})
        if entity_pages:
            wiki_contexts = entity_pages.get("wiki_context", [])
            for ctx in wiki_contexts:
                if ctx and _check_evidence_present(ctx, all_aliases):
                    # 길이 제한 적용
                    context = ctx[:max_context_length] if len(ctx) > max_context_length else ctx
                    break

        # entity_pages에서 못 찾으면 search_results 탐색
        if context is None:
            search_results = item.get("search_results", {})
            if search_results:
                search_contexts = search_results.get("search_context", [])
                for ctx in search_contexts:
                    if ctx and _check_evidence_present(ctx, all_aliases):
                        context = ctx[:max_context_length] if len(ctx) > max_context_length else ctx
                        break

        if context is None:
            skipped_no_context += 1
            continue

        # evidence_present 최종 확인
        evidence_present = _check_evidence_present(context, all_aliases)
        if not evidence_present:
            skipped_no_evidence += 1
            continue

        processed.append({
            "question_id": item.get("question_id", f"tqa_{idx}"),
            "question": question,
            "answer_value": answer_value,
            "answer_aliases": all_aliases,
            "context": context,
            "evidence_present": evidence_present,
        })

    logger.info(
        f"Processed: {len(processed)} items "
        f"(skipped: {skipped_no_context} no context, "
        f"{skipped_no_evidence} no evidence)"
    )
    return processed


def save_processed_data(data: list[dict], output_path: str) -> None:
    """Save processed data to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} items to {output}")


def load_processed_data(input_path: str) -> list[dict]:
    """Load processed data from JSONL."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} items from {input_path}")
    return data

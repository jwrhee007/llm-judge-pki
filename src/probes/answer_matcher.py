"""
Answer Matcher for Knowledge Probe.

Probe 응답을 gold answer와 비교하여 정답 여부를 판단한다.
1차: String matching (alias 기반, 대소문자/공백 무시)
2차: LLM-based matching (string match 실패 시 fallback)
"""

import re

from src.api.openai_client import OpenAIClient
from src.prompts.probe_prompts import build_answer_match_messages
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles/punctuation."""
    text = text.lower().strip()
    # Remove leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def string_match(
    predicted: str,
    gold_answer: str,
    aliases: list[str],
) -> bool:
    """
    String-based answer matching.

    predicted answer가 gold answer 또는 alias 중 하나와 일치하면 True.
    부분 포함(containment) 방식으로 검사한다.
    """
    norm_pred = _normalize(predicted)
    if not norm_pred:
        return False

    all_answers = [gold_answer] + (aliases or [])
    for ans in all_answers:
        norm_ans = _normalize(ans)
        if not norm_ans:
            continue
        # Exact match
        if norm_pred == norm_ans:
            return True
        # Containment: predicted에 gold가 포함되거나 그 반대
        if norm_ans in norm_pred or norm_pred in norm_ans:
            return True

    return False


def llm_match(
    client: OpenAIClient,
    question: str,
    gold_answer: str,
    aliases: list[str],
    predicted: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 8,
) -> bool:
    """
    LLM-based answer equivalence check.

    String matching으로 판별이 어려운 경우 LLM을 사용하여
    의미적 동등성을 판단한다.
    """
    messages = build_answer_match_messages(
        question=question,
        gold_answer=gold_answer,
        aliases=aliases,
        predicted_answer=predicted,
    )

    try:
        response = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(f"LLM match failed: {e}")
        return False


def match_answer(
    client: OpenAIClient | None,
    question: str,
    gold_answer: str,
    aliases: list[str],
    predicted: str,
    model: str = "gpt-4o-mini-2024-07-18",
    use_llm_fallback: bool = True,
) -> dict:
    """
    Gold answer와 predicted answer의 일치 여부를 판단한다.

    Returns:
        dict with keys:
            - is_correct: bool
            - match_method: "string" | "llm" | "no_match"
    """
    # 빈 응답 처리
    if not predicted or not predicted.strip():
        return {"is_correct": False, "match_method": "empty"}

    # "I don't know" 류의 응답 필터링
    refusal_patterns = [
        r"i don'?t know",
        r"i'?m not sure",
        r"i cannot",
        r"i can'?t",
        r"i do not know",
        r"i am not sure",
        r"i'm unable",
        r"i am unable",
        r"no answer",
        r"unknown",
    ]
    lower_pred = predicted.lower().strip()
    for pattern in refusal_patterns:
        if re.search(pattern, lower_pred) and len(lower_pred) < 100:
            return {"is_correct": False, "match_method": "refusal"}

    # 1차: String matching
    if string_match(predicted, gold_answer, aliases):
        return {"is_correct": True, "match_method": "string"}

    # 2차: LLM fallback
    if use_llm_fallback and client is not None:
        is_correct = llm_match(
            client=client,
            question=question,
            gold_answer=gold_answer,
            aliases=aliases,
            predicted=predicted,
            model=model,
        )
        return {
            "is_correct": is_correct,
            "match_method": "llm" if is_correct else "no_match",
        }

    return {"is_correct": False, "match_method": "no_match"}

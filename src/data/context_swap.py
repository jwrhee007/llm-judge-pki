"""
Context Swap Module.

Same-type 및 Cross-type context swap을 수행한다.

Same-type swap: 같은 NER 태그의 다른 문항 context로 교체
  → 주제적 유사성이 있어 Judge가 과잉 해석할 여지 → PKI 발동 유도
  → Lee et al.의 Type-Preserving (TP) swap에 대응

Cross-type swap: 다른 NER 태그의 문항 context로 교체
  → 주제가 완전히 달라 "evidence 없음"을 쉽게 판단 → 통제 조건
  → Lee et al.의 Type-Changing (TC) swap에 대응

핵심 제약:
  - Swapped context에 원래 answer가 포함되면 안 됨
    (우연히 evidence가 있으면 PKI 측정이 오염)
"""

import random
from collections import defaultdict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _answer_in_context(answer_aliases: list[str], context: str) -> bool:
    """Answer alias 중 하나라도 context에 포함되어 있으면 True."""
    ctx_lower = context.lower()
    for alias in answer_aliases:
        if alias.lower() in ctx_lower:
            return True
    return False


def build_swap_pairs(
    data: list[dict],
    swap_type: str = "same",
    ner_tag_key: str = "ner_tag",
    seed: int = 42,
    max_retries: int = 50,
) -> list[dict]:
    """
    각 문항에 대해 swap partner를 결정하고, swapped context를 할당한다.

    Args:
        data: sampled data (question, context, answer_value, answer_aliases, ner_tag)
        swap_type: "same" (Same-type) or "cross" (Cross-type)
        ner_tag_key: NER 태그 필드명
        seed: 랜덤 시드
        max_retries: answer-in-context 충돌 시 재시도 횟수

    Returns:
        list of dict — 원본 데이터에 다음 필드 추가:
          - context_swap: swapped context
          - swap_partner_idx: swap partner의 data index
          - swap_partner_qid: swap partner의 question_id
          - swap_type: "same" or "cross"
          - swap_valid: bool (answer가 swapped context에 없으면 True)
    """
    rng = random.Random(seed)

    # NER 태그별 인덱스 그룹핑
    tag_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(data):
        tag = item.get(ner_tag_key, "UNKNOWN")
        tag_to_indices[tag].append(idx)

    # 태그 목록
    all_tags = sorted(tag_to_indices.keys())
    all_indices = list(range(len(data)))

    swap_success = 0
    swap_fail = 0
    swap_no_candidate = 0

    for idx, item in enumerate(data):
        tag = item.get(ner_tag_key, "UNKNOWN")
        answer_aliases = item.get("answer_aliases", [item["answer_value"]])

        if swap_type == "same":
            # Same-type: 같은 NER 태그의 다른 문항
            candidates = [i for i in tag_to_indices[tag] if i != idx]
        elif swap_type == "cross":
            # Cross-type: 다른 NER 태그의 문항
            other_tags = [t for t in all_tags if t != tag]
            candidates = []
            for t in other_tags:
                candidates.extend(tag_to_indices[t])
        else:
            raise ValueError(f"Unknown swap_type: {swap_type}")

        if not candidates:
            # 후보가 없는 경우 (희소 태그)
            item["context_swap"] = None
            item["swap_partner_idx"] = -1
            item["swap_partner_qid"] = None
            item["swap_type"] = swap_type
            item["swap_valid"] = False
            swap_no_candidate += 1
            continue

        # Answer가 swapped context에 포함되지 않는 partner 찾기
        rng.shuffle(candidates)
        found = False
        for attempt, partner_idx in enumerate(candidates):
            if attempt >= max_retries:
                break
            partner_context = data[partner_idx]["context"]
            if not _answer_in_context(answer_aliases, partner_context):
                item["context_swap"] = partner_context
                item["swap_partner_idx"] = partner_idx
                item["swap_partner_qid"] = data[partner_idx]["question_id"]
                item["swap_type"] = swap_type
                item["swap_valid"] = True
                swap_success += 1
                found = True
                break

        if not found:
            # 모든 후보에 answer가 포함되는 경우 (매우 드묾)
            # 첫 번째 후보를 사용하되 invalid로 표시
            partner_idx = candidates[0]
            item["context_swap"] = data[partner_idx]["context"]
            item["swap_partner_idx"] = partner_idx
            item["swap_partner_qid"] = data[partner_idx]["question_id"]
            item["swap_type"] = swap_type
            item["swap_valid"] = False
            swap_fail += 1

    logger.info(
        f"Context swap ({swap_type}): "
        f"success={swap_success}, "
        f"fail(answer in swap)={swap_fail}, "
        f"no_candidate={swap_no_candidate}, "
        f"total={len(data)}"
    )

    return data


def get_swap_stats(data: list[dict]) -> dict:
    """Swap 결과 통계를 반환한다."""
    valid = sum(1 for d in data if d.get("swap_valid", False))
    invalid = sum(1 for d in data if not d.get("swap_valid", True))
    no_swap = sum(1 for d in data if d.get("context_swap") is None)

    return {
        "total": len(data),
        "valid": valid,
        "invalid": invalid,
        "no_swap": no_swap,
        "valid_ratio": valid / len(data) if data else 0,
    }

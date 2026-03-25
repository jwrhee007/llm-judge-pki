"""
NER 태그별 층화 추출 (Stratified Sampling).

NER 태깅된 문항에서 태그별 최대 20개씩 추출하여
~280-300 문항의 균형 잡힌 실험 데이터셋을 구성한다.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def stratified_sample(
    data: list[dict],
    ner_tag_key: str = "ner_tag",
    max_per_tag: int = 20,
    target_total: int = 288,
    seed: int = 42,
    target_tags: list[str] | None = None,
) -> list[dict]:
    """
    NER 태그별 층화 추출을 수행한다.

    Args:
        data: NER 태그가 포함된 데이터 리스트
        ner_tag_key: NER 태그 필드명
        max_per_tag: 태그별 최대 추출 수
        target_total: 목표 총 문항 수
        seed: 랜덤 시드
        target_tags: 대상 NER 태그 리스트 (None이면 전체 사용)

    Returns:
        층화 추출된 데이터 리스트
    """
    rng = random.Random(seed)

    # NER 태그별 그룹핑
    tag_groups: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        tag = item.get(ner_tag_key, "UNKNOWN")
        if target_tags and tag not in target_tags:
            continue
        if tag in ("UNKNOWN", "NAN"):
            continue
        tag_groups[tag].append(item)

    logger.info(f"NER tag distribution (before sampling):")
    for tag, items in sorted(tag_groups.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {tag}: {len(items)}")

    # 각 태그에서 최대 max_per_tag개 추출
    sampled = []
    tag_counts = {}
    for tag, items in sorted(tag_groups.items()):
        n_sample = min(len(items), max_per_tag)
        selected = rng.sample(items, n_sample)
        sampled.extend(selected)
        tag_counts[tag] = n_sample

    # target_total 초과 시 비례적으로 축소
    if len(sampled) > target_total:
        sampled = rng.sample(sampled, target_total)
        # 재집계
        tag_counts = Counter(item[ner_tag_key] for item in sampled)

    logger.info(f"Sampled {len(sampled)} items across {len(tag_counts)} NER tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {tag}: {count}")

    return sampled


def save_sampled_data(data: list[dict], output_path: str) -> None:
    """Save sampled data to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} sampled items to {output}")


def load_sampled_data(input_path: str) -> list[dict]:
    """Load sampled data from JSONL."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} sampled items from {input_path}")
    return data

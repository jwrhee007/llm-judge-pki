"""
Sampling utilities for experiment data selection.

Two strategies:
  1. Random sampling (Lee et al. 방식): 전체 데이터에서 무작위 추출
  2. Stratified sampling: NER 태그별 균등 추출

사후 층화 분석(post-hoc NER analysis)은 random sampling 후
NER 태그별 분포를 확인하고, 충분한 표본이 있는 태그만 개별 분석,
나머지는 OTHER로 묶는다.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------------------------------
# Random sampling (Lee et al. 방식)
# -----------------------------------------------------------------------

def random_sample(
    data: list[dict],
    target_total: int = 1000,
    seed: int = 42,
    ner_tag_key: str = "ner_tag",
    exclude_tags: list[str] | None = None,
) -> list[dict]:
    """
    전체 데이터에서 무작위 추출 (Lee et al. 방식).

    UNKNOWN/NAN 태그는 제외한다.
    데이터가 target_total보다 적으면 전체를 사용한다.

    Args:
        data: NER 태그가 포함된 데이터 리스트
        target_total: 목표 총 문항 수
        seed: 랜덤 시드
        ner_tag_key: NER 태그 필드명
        exclude_tags: 제외할 NER 태그 리스트

    Returns:
        랜덤 추출된 데이터 리스트
    """
    rng = random.Random(seed)
    exclude = set(exclude_tags or []) | {"UNKNOWN", "NAN"}

    # 유효한 태그를 가진 데이터만 필터링
    valid_data = [
        item for item in data
        if item.get(ner_tag_key, "UNKNOWN") not in exclude
    ]

    logger.info(f"Valid items (excluding {exclude}): {len(valid_data)} / {len(data)}")

    # 랜덤 샘플링
    if len(valid_data) <= target_total:
        sampled = valid_data
        logger.info(f"Data ({len(valid_data)}) <= target ({target_total}), using all")
    else:
        sampled = rng.sample(valid_data, target_total)

    # NER 태그 분포 로깅
    tag_counts = Counter(item[ner_tag_key] for item in sampled)
    logger.info(f"Sampled {len(sampled)} items, NER tag distribution:")
    for tag, count in tag_counts.most_common():
        logger.info(f"  {tag}: {count}")

    return sampled


# -----------------------------------------------------------------------
# Stratified sampling (기존 방식)
# -----------------------------------------------------------------------

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
        tag_counts = Counter(item[ner_tag_key] for item in sampled)

    logger.info(f"Sampled {len(sampled)} items across {len(tag_counts)} NER tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {tag}: {count}")

    return sampled


# -----------------------------------------------------------------------
# Post-hoc NER analysis utilities
# -----------------------------------------------------------------------

def posthoc_ner_summary(
    data: list[dict],
    ner_tag_key: str = "ner_tag",
    min_count: int = 15,
) -> dict:
    """
    사후 층화 분석을 위한 NER 태그 요약.

    min_count 이상인 태그는 개별 분석 대상으로,
    미만인 태그는 OTHER로 묶는다.

    Args:
        data: NER 태그가 포함된 데이터 리스트
        min_count: 개별 분석 최소 표본 수
        ner_tag_key: NER 태그 필드명

    Returns:
        dict with:
            - analyzable_tags: list[str] — 개별 분석 가능 태그
            - other_tags: list[str] — OTHER로 묶이는 태그
            - tag_counts: dict[str, int] — 전체 태그별 빈도
            - tag_mapping: dict[str, str] — 원래 태그 → 분석용 태그 매핑
    """
    tag_counts = Counter(item.get(ner_tag_key, "UNKNOWN") for item in data)

    analyzable_tags = []
    other_tags = []

    for tag, count in tag_counts.most_common():
        if tag in ("UNKNOWN", "NAN"):
            other_tags.append(tag)
        elif count >= min_count:
            analyzable_tags.append(tag)
        else:
            other_tags.append(tag)

    # 매핑 생성
    tag_mapping = {}
    for tag in analyzable_tags:
        tag_mapping[tag] = tag
    for tag in other_tags:
        tag_mapping[tag] = "OTHER"

    logger.info(f"Post-hoc NER analysis (min_count={min_count}):")
    logger.info(f"  Analyzable tags ({len(analyzable_tags)}):")
    for tag in analyzable_tags:
        logger.info(f"    {tag}: {tag_counts[tag]}")
    other_total = sum(tag_counts[t] for t in other_tags)
    if other_tags:
        logger.info(f"  OTHER ({len(other_tags)} tags, {other_total} items):")
        for tag in other_tags:
            logger.info(f"    {tag}: {tag_counts[tag]}")

    return {
        "analyzable_tags": analyzable_tags,
        "other_tags": other_tags,
        "tag_counts": dict(tag_counts),
        "tag_mapping": tag_mapping,
    }


def assign_analysis_tag(
    data: list[dict],
    tag_mapping: dict[str, str],
    ner_tag_key: str = "ner_tag",
    analysis_tag_key: str = "analysis_tag",
) -> list[dict]:
    """
    사후 분석용 태그를 데이터에 추가한다.
    min_count 미만 태그는 OTHER로 매핑된다.
    """
    for item in data:
        original_tag = item.get(ner_tag_key, "UNKNOWN")
        item[analysis_tag_key] = tag_mapping.get(original_tag, "OTHER")
    return data


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

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
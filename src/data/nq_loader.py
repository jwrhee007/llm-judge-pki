"""
Natural Questions (full) 데이터 로딩.

NQ는 Google 검색 쿼리 + Wikipedia 페이지로 구성되며,
annotator가 long_answer (paragraph)와 short_answer (entity span)를 표시한다.

short_answer가 long_answer 내에 포함되므로 evidence grounding이 보장된다.

HuggingFace NQ 데이터 구조 (validation, 5-way annotated):
  annotations:
    long_answer: list of dict (5개) — each has start_token, end_token, candidate_index
    short_answers: list of dict (5개) — each has start_token[], end_token[], text[]
    yes_no_answer: list of int (5개)
  document:
    tokens: dict with token[], is_html[]
  question:
    text: str
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _tokens_to_text(token_list: list[str], is_html_list: list[bool], start: int, end: int) -> str:
    """
    토큰 리스트에서 [start, end) 범위의 텍스트를 추출한다.
    HTML 토큰은 건너뛰고 텍스트만 연결한다.
    """
    selected = []
    for i in range(start, min(end, len(token_list))):
        if not is_html_list[i]:
            selected.append(token_list[i])
    return " ".join(selected)


def _extract_from_item(item: dict) -> dict | None:
    """
    NQ item에서 (question, context, answer)를 추출한다.

    Validation split은 5-way annotated이므로,
    long_answer와 short_answer가 모두 유효한 첫 번째 annotation을 사용한다.

    Returns:
        dict with question, context, answer_value, answer_aliases or None
    """
    annotations = item.get("annotations", {})
    long_answers = annotations.get("long_answer", [])
    short_answers_list = annotations.get("short_answers", [])

    # Token lists
    doc_tokens = item.get("document", {}).get("tokens", {})
    token_list = doc_tokens.get("token", [])
    is_html_list = doc_tokens.get("is_html", [])

    if not token_list:
        return None

    # Iterate over annotations to find first valid one
    n_annotations = len(long_answers)

    for ann_idx in range(n_annotations):
        la = long_answers[ann_idx]
        la_start = la.get("start_token", -1)
        la_end = la.get("end_token", -1)

        # Skip null long answers
        if la_start < 0 or la_end < 0 or la_start >= la_end:
            continue

        # Get short answers for this annotation
        sa = short_answers_list[ann_idx]
        sa_starts = sa.get("start_token", [])
        sa_ends = sa.get("end_token", [])
        sa_texts = sa.get("text", [])

        # Skip if no short answers
        if not sa_starts or not sa_ends:
            continue

        # Extract context (long answer paragraph, HTML removed)
        context = _tokens_to_text(token_list, is_html_list, la_start, la_end)
        if not context or len(context.strip()) < 20:
            continue

        # Extract answer (first short answer span)
        if sa_texts and sa_texts[0]:
            answer_value = sa_texts[0]
        else:
            answer_value = _tokens_to_text(
                token_list, is_html_list, sa_starts[0], sa_ends[0]
            )

        if not answer_value or not answer_value.strip():
            continue

        # Build aliases from all short answer spans
        aliases = set()
        aliases.add(answer_value.strip())
        for i in range(len(sa_starts)):
            if sa_texts and i < len(sa_texts) and sa_texts[i]:
                aliases.add(sa_texts[i].strip())
            else:
                sa_text = _tokens_to_text(
                    token_list, is_html_list, sa_starts[i], sa_ends[i]
                )
                if sa_text:
                    aliases.add(sa_text.strip())

        # Verify answer is in context (should always be true)
        if answer_value.strip().lower() not in context.lower():
            continue

        return {
            "context": context.strip(),
            "answer_value": answer_value.strip(),
            "answer_aliases": list(aliases),
        }

    return None


def load_nq_full(
    split: str = "validation",
    max_context_length: int = 3000,
) -> list[dict]:
    """
    NQ (full) 데이터셋을 로드하고 (question, context, answer) 트리플렛을 추출한다.

    Args:
        split: "validation" (7,830 examples, 5-way annotated)
        max_context_length: context 최대 문자 수

    Returns:
        list of dict with question_id, question, answer_value, answer_aliases,
        context, evidence_present (always True for NQ)
    """
    logger.info(f"Loading NQ (full) ({split})...")
    logger.info("Note: First download may take a while (~42GB for train, ~1GB for validation)")

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split=split,
    )
    logger.info(f"Raw dataset size: {len(ds)}")

    processed = []
    skipped_no_answer = 0
    skipped_too_long = 0

    for idx, item in enumerate(ds):
        # Extract question
        question = item.get("question", {})
        if isinstance(question, dict):
            question_text = question.get("text", "")
        else:
            question_text = str(question)

        if not question_text:
            continue

        # Extract (context, answer) from annotations
        extraction = _extract_from_item(item)
        if extraction is None:
            skipped_no_answer += 1
            continue

        context = extraction["context"]
        if len(context) > max_context_length:
            context = context[:max_context_length]
            skipped_too_long += 1

        # Question ID
        qid = item.get("id", f"nq_{idx}")
        if isinstance(qid, (bytes, bytearray)):
            qid = qid.decode("utf-8")
        qid = str(qid)

        processed.append({
            "question_id": qid,
            "question": question_text,
            "answer_value": extraction["answer_value"],
            "answer_aliases": extraction["answer_aliases"],
            "context": context,
            "evidence_present": True,  # Guaranteed by NQ annotation
        })

        # Progress logging
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(ds)} items...")

    logger.info(
        f"Processed: {len(processed)} items "
        f"(skipped: {skipped_no_answer} no valid answer, "
        f"{skipped_too_long} truncated)"
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
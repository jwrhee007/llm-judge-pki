"""
Knowledge Probe Prompts — Exp. 2-0.

Judge에게 context 없이 질문만 제시하여 사전지식 보유 여부를 확인한다.

방식 A (Bare question): 질문만 제시
방식 B (Knowledge-eliciting): parametric knowledge 활용을 명시적으로 유도

Lee et al. (2026)의 Evaluator-Knowledge Swap에 대응하는 실험 단계.
"""


def build_probe_messages_method_a(question: str) -> list[dict]:
    """
    방식 A — Bare question.

    질문만 제시한다. 모델이 확신 없으면 "I don't know"로 빠질 수 있어
    knows rate를 과소 추정할 수 있다.
    """
    return [
        {
            "role": "user",
            "content": question,
        }
    ]


def build_probe_messages_method_b(question: str) -> list[dict]:
    """
    방식 B — Knowledge-eliciting prompt.

    Parametric knowledge 활용을 명시적으로 유도한다.
    모델이 강제로 답변하므로 hallucination이 증가할 수 있어
    knows rate를 과대 추정할 수 있다.
    Self-consistency 3회가 이 차이를 완화한다.
    """
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question using only your internal knowledge.\n"
                'Do not say "I don\'t know". Give your best answer.\n'
                f"Q: {question}"
            ),
        }
    ]


# -----------------------------------------------------------------------
# Answer matching prompt — Probe 응답과 gold answer 비교
# -----------------------------------------------------------------------

ANSWER_MATCH_SYSTEM_PROMPT = (
    "You are an answer equivalence checker. Given a question, a gold answer "
    "(with possible aliases), and a predicted answer, determine if the predicted "
    "answer is semantically equivalent to the gold answer.\n"
    "Return ONLY 'YES' or 'NO'."
)

ANSWER_MATCH_USER_TEMPLATE = """\
Question: {question}
Gold answer: {gold_answer}
Gold aliases: {aliases}
Predicted answer: {predicted_answer}

Is the predicted answer semantically equivalent to the gold answer? \
Return ONLY 'YES' or 'NO'."""


def build_answer_match_messages(
    question: str,
    gold_answer: str,
    aliases: list[str],
    predicted_answer: str,
) -> list[dict]:
    """Build messages for LLM-based answer equivalence checking."""
    return [
        {"role": "system", "content": ANSWER_MATCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_MATCH_USER_TEMPLATE.format(
                question=question,
                gold_answer=gold_answer,
                aliases=", ".join(aliases) if aliases else gold_answer,
                predicted_answer=predicted_answer,
            ),
        },
    ]

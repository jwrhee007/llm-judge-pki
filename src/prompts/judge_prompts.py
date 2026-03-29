"""
Judge Prompts — Context-Grounded Adaptation of Lee et al. (2026).

Lee et al.의 SimpleQA 프롬프트(Table 9, 10, 11)를 context-grounded QA 평가로 변환.
"gold target"을 "context passage"로, verdict 기준을 context 기반으로 변경.

§7.2 변환 규칙:
  - CORRECT → context에 의해 supported
  - INCORRECT → context에 의해 contradicted 또는 not supported
  - NOT_ATTEMPTED → context에 insufficient information
"""

# =====================================================================
# P-Lee-Standard — Lee et al. Table 9의 context-grounded 변환
# =====================================================================

P_LEE_STANDARD_TEMPLATE = """\
Your job is to look at a question, a context passage, and a candidate answer, and then determine \
if the candidate answer is supported by the context passage. Assign a grade of either \
["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: sasha and malia obama
Candidate answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Candidate answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These candidate answers are all CORRECT because:
- They fully contain the important information supported by the context passage.
- They do not contain any information that contradicts the context passage.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the answer is fully supported by the context and contains no incorrect information or contradictions.

The following are examples of INCORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: Malia.
Candidate answer 2: Malia, Sasha, and Susan.
Candidate answer 3: Barack Obama does not have any children.
Candidate answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Candidate answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Candidate answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These candidate answers are all INCORRECT because:
- A factual statement in the answer contradicts the context passage. Incorrect statements that have some hedging (e.g., "it is possible that", "although I'm not sure, I think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama served as the 44th President of the United States from 2009 to 2017.
Candidate answer 1: I don't know.
Candidate answer 2: I need more context about which Obama you are talking about.
Candidate answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama was a president.
Candidate answer 4: Barack Obama was president. I know that he has children, but I'm not sure about their names.
```
These candidate answers are all NOT_ATTEMPTED because:
- The context passage does not contain sufficient information to answer the question.
- The important information needed to answer the question is not present in the context.
- No statements in the answer contradict the context passage.

Also note the following things:
- For grading questions where the answer is a number, the candidate answer needs to be correct to the last significant figure as supported by the context. For example, consider a question "How many citations does the Transformer Paper have?" with context stating "The Transformer paper has been cited approximately 120,000 times."
  - Candidate answers "120k", "124k", and "115k" are all CORRECT.
  - Candidate answers "100k" and "113k" are INCORRECT.
  - Candidate answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the context.
- The context passage may contain more information than needed. The candidate answer only needs to contain the information relevant to the question.
- Do not punish candidate answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's names if it's clearly the same name.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. \
Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

```
Question: {question}
Context passage: {context}
Candidate answer: {candidate_answer}
```

Grade the candidate answer of this new question as one of:
A: CORRECT B: INCORRECT C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""


# =====================================================================
# P-Lee-Direct — Lee et al. Table 10의 context-grounded 변환
# =====================================================================

P_LEE_DIRECT_TEMPLATE = """\
Your job is to look at a question, a context passage, and a candidate answer, and then determine \
if the candidate answer is supported by the context passage. Assign a grade of either \
["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

IMPORTANT: The context passage is the ONLY source of truth. You MUST evaluate the candidate answer \
based ONLY on the given context passage and the question. Do NOT use outside knowledge, web access, \
or personal beliefs to override the context passage. Even if you believe the candidate answer is \
factually correct based on your own knowledge, you must grade it as NOT_ATTEMPTED if the context \
passage does not contain sufficient information to support it.

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: sasha and malia obama
Candidate answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Candidate answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These candidate answers are all CORRECT because:
- They fully contain the important information supported by the context passage.
- They do not contain any information that contradicts the context passage.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the answer is fully supported by the context and contains no incorrect information or contradictions.

The following are examples of INCORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: Malia.
Candidate answer 2: Malia, Sasha, and Susan.
Candidate answer 3: Barack Obama does not have any children.
Candidate answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Candidate answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Candidate answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These candidate answers are all INCORRECT because:
- A factual statement in the answer contradicts the context passage. Incorrect statements that have some hedging (e.g., "it is possible that", "although I'm not sure, I think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama served as the 44th President of the United States from 2009 to 2017.
Candidate answer 1: I don't know.
Candidate answer 2: I need more context about which Obama you are talking about.
Candidate answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama was a president.
Candidate answer 4: Barack Obama was president. I know that he has children, but I'm not sure about their names.
```
These candidate answers are all NOT_ATTEMPTED because:
- The context passage does not contain sufficient information to answer the question.
- The important information needed to answer the question is not present in the context.
- No statements in the answer contradict the context passage.

Also note the following things:
- For grading questions where the answer is a number, the candidate answer needs to be correct to the last significant figure as supported by the context. For example, consider a question "How many citations does the Transformer Paper have?" with context stating "The Transformer paper has been cited approximately 120,000 times."
  - Candidate answers "120k", "124k", and "115k" are all CORRECT.
  - Candidate answers "100k" and "113k" are INCORRECT.
  - Candidate answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the context.
- The context passage may contain more information than needed. The candidate answer only needs to contain the information relevant to the question.
- Do not punish candidate answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's names if it's clearly the same name.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. \
Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

```
Question: {question}
Context passage: {context}
Candidate answer: {candidate_answer}
```

Grade the candidate answer of this new question as one of:
A: CORRECT B: INCORRECT C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""


# =====================================================================
# P-Lee-CoT — Lee et al. Table 11의 context-grounded 변환
# =====================================================================

P_LEE_COT_TEMPLATE = """\
Your job is to look at a question, a context passage, and a candidate answer, and then determine \
if the candidate answer is supported by the context passage. Assign a grade of either \
["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

First, I will give examples of each grade, and then you will grade a new example.

IMPORTANT OUTPUT INSTRUCTIONS:
- You MUST show your reasoning before giving the final grade.
- Output MUST have exactly two parts, in this order:
  1) A section starting with "Reasoning:" where you explain step-by-step whether the candidate answer is supported by the context passage.
  2) A final line starting with "Final:" followed by ONLY ONE letter: "A", "B", or "C".
- Keep the "Reasoning:" concise (at most 5 bullet points). Do not include any other text after the "Final:" line.

The following are examples of CORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: sasha and malia obama
Candidate answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Candidate answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These candidate answers are all CORRECT because:
- They fully contain the important information supported by the context passage.
- They do not contain any information that contradicts the context passage.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the answer is fully supported by the context and contains no incorrect information or contradictions.

The following are examples of INCORRECT candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama and his wife Michelle have two daughters, Malia Ann and Natasha Marian, commonly known as Malia and Sasha Obama.
Candidate answer 1: Malia.
Candidate answer 2: Malia, Sasha, and Susan.
Candidate answer 3: Barack Obama does not have any children.
Candidate answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Candidate answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Candidate answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These candidate answers are all INCORRECT because:
- A factual statement in the answer contradicts the context passage. Incorrect statements that have some hedging (e.g., "it is possible that", "although I'm not sure, I think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED candidate answers.
```
Question: What are the names of Barack Obama's children?
Context passage: Barack Obama served as the 44th President of the United States from 2009 to 2017.
Candidate answer 1: I don't know.
Candidate answer 2: I need more context about which Obama you are talking about.
Candidate answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama was a president.
Candidate answer 4: Barack Obama was president. I know that he has children, but I'm not sure about their names.
```
These candidate answers are all NOT_ATTEMPTED because:
- The context passage does not contain sufficient information to answer the question.
- The important information needed to answer the question is not present in the context.
- No statements in the answer contradict the context passage.

Also note the following things:
- For grading questions where the answer is a number, the candidate answer needs to be correct to the last significant figure as supported by the context. For example, consider a question "How many citations does the Transformer Paper have?" with context stating "The Transformer paper has been cited approximately 120,000 times."
  - Candidate answers "120k", "124k", and "115k" are all CORRECT.
  - Candidate answers "100k" and "113k" are INCORRECT.
  - Candidate answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the context.
- The context passage may contain more information than needed. The candidate answer only needs to contain the information relevant to the question.
- Do not punish candidate answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's names if it's clearly the same name.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. \
Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

```
Question: {question}
Context passage: {context}
Candidate answer: {candidate_answer}
```

Grade the candidate answer of this new question as one of:
A: CORRECT B: INCORRECT C: NOT_ATTEMPTED
Remember: output "Reasoning:" then "Final: <A/B/C>"""


# =====================================================================
# Prompt registry and helpers
# =====================================================================

PROMPT_REGISTRY = {
    "P-Lee-Standard": P_LEE_STANDARD_TEMPLATE,
    "P-Lee-Direct": P_LEE_DIRECT_TEMPLATE,
    "P-Lee-CoT": P_LEE_COT_TEMPLATE,
}

VERDICT_MAP = {
    "A": "CORRECT",
    "B": "INCORRECT",
    "C": "NOT_ATTEMPTED",
}


def build_judge_messages(
    question: str,
    context: str,
    candidate_answer: str,
    prompt_id: str = "P-Lee-Standard",
) -> list[dict]:
    """Build OpenAI messages for judge evaluation."""
    template = PROMPT_REGISTRY.get(prompt_id)
    if template is None:
        raise ValueError(
            f"Unknown prompt_id: {prompt_id}. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )

    content = template.format(
        question=question,
        context=context,
        candidate_answer=candidate_answer,
    )

    return [{"role": "user", "content": content}]


def parse_verdict(response: str, prompt_id: str = "P-Lee-Standard") -> str:
    """
    Parse judge response to extract verdict.

    Returns one of: CORRECT, INCORRECT, NOT_ATTEMPTED, PARSE_ERROR
    """
    text = response.strip()

    if prompt_id == "P-Lee-CoT":
        # CoT: extract verdict after "Final:"
        lines = text.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.lower().startswith("final:"):
                text = line.split(":", 1)[1].strip()
                break

    # Extract letter (A/B/C)
    text = text.strip().upper()

    # Direct letter match
    if text in VERDICT_MAP:
        return VERDICT_MAP[text]

    # First character match
    if text and text[0] in VERDICT_MAP:
        return VERDICT_MAP[text[0]]

    # Keyword match
    text_lower = response.strip().lower()
    if "not_attempted" in text_lower or "not attempted" in text_lower:
        return "NOT_ATTEMPTED"
    if "incorrect" in text_lower:
        return "INCORRECT"
    if "correct" in text_lower:
        return "CORRECT"

    return "PARSE_ERROR"

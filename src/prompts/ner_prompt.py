"""
NER Prompt — Lee et al. (2026), Figure 6.

Reference answer에 대해 SpaCy NER 태그를 할당하는 프롬프트.
본 연구에서는 TriviaQA의 gold answer에 대해 NER 태그를 부여하여
NER 태그별 층화 추출에 사용한다.
"""

NER_SYSTEM_PROMPT = (
    "You are an information extraction model that assigns exactly ONE SpaCy NER type to the given original answer. Return ONLY the label token. No extra text."
)

NER_USER_PROMPT_TEMPLATE = """\
## SpaCy NER labels & definitions
PERSON: People, including fictional.
NORP: Nationalities or religious or political groups.
FAC: Buildings, airports, highways, bridges, etc.
ORG: Companies, agencies, institutions, etc.
GPE: Countries, cities, states.
LOC: Non-GPE locations, mountain ranges, bodies of water.
PRODUCT: Objects, vehicles, foods, devices (not services).
EVENT: Named hurricanes, battles, wars, sports events, festivals, etc.
WORK_OF_ART: Titles of books, movies, songs, paintings, etc.
LAW: Named documents made into laws.
LANGUAGE: Any named language.
DATE: Absolute or relative dates or periods.
TIME: Times smaller than a day.
PERCENT: Percentage, including "%".
MONEY: Monetary values, including unit.
QUANTITY: Measurements of size/weight/distance/volume/speed/etc.
ORDINAL: "first", "second", "23rd", etc.
CARDINAL: Numerals that do not fall under another type.

## Rules
1) Classify the answer string AS WRITTEN (no external lookup).
2) If multiple entities appear, label by the main head of the answer.
3) Numeric answers: - With unit → QUANTITY (e.g., 5 km), MONEY (e.g., €10), PERCENT (e.g., 12%), TIME (e.g., 3 hours). - Dates/periods → DATE. - Ordinals → ORDINAL. - Plain counts/integers → CARDINAL.
4) GPE vs LOC: Countries/cities/states → GPE; geographic features → LOC.
5) ORG vs PRODUCT: Organizations → ORG; tangible items → PRODUCT.
6) WORK_OF_ART only for titled creative works.
7) Output exactly one label from the list above. No explanation.
8) If answer do not have an entity just return NAN

# Few-shot examples
Q: Who wrote *Pride and Prejudice? Original Answer: Jane Austen Label: PERSON
Q: What is the capital of France? Original Answer: Paris Label: GPE
Q: Which company makes the iPhone? Original Answer: Apple Label: ORG
Q: What language is primarily spoken in Brazil? Original Answer: Portuguese Label: LANGUAGE
Q: When did WW2 end? Original Answer: 1945 Label: DATE
Q: How long is a marathon? Original Answer: 42.195 km Label: QUANTITY
Q: Which event did the Chiefs win in 2024? Original Answer: Super Bowl LVIII Label: EVENT
Q: What is "Mona Lisa"? Original Answer: Mona Lisa Label: WORK_OF_ART
Q: Where is Mount Everest? Original Answer: Himalayas Label: LOC
Q: How much does it cost? Original Answer: $20 Label: MONEY
Q: What is John Mayne's occupation? Original Answer: journalist Label: NAN

# Task
Question: {question}
Original Answer: {answer}

# Output
Return ONLY the label (one of: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL, NAN)."""


VALID_NER_TAGS = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
    "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "NAN",
]


def build_ner_messages(question: str, answer: str) -> list[dict]:
    """Build OpenAI messages for NER tagging."""
    return [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": NER_USER_PROMPT_TEMPLATE.format(
                question=question, answer=answer
            ),
        },
    ]


def parse_ner_response(response: str) -> str:
    """Parse and validate NER response. Returns tag or 'UNKNOWN'."""
    tag = response.strip().upper().replace('"', "").replace("'", "")
    # Handle cases where model returns extra text
    for valid_tag in VALID_NER_TAGS:
        if valid_tag in tag:
            return valid_tag
    return "UNKNOWN"

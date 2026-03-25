# LLM-as-Judge PKI (Parametric Knowledge Interference) Study

LLM-as-a-Judge가 context-grounded QA 평가에서 제공된 context에 충실하게 판정하는지,  
아니면 내부 parametric knowledge에 의존하여 판정을 왜곡하는지 체계적으로 검증하는 연구 코드베이스.

Lee et al. (2026)의 swapped-reference QA 프레임워크를 context 차원으로 확장합니다.

## 프로젝트 구조

```
llm-judge-pki/
├── configs/
│   └── config.yaml              # 실험 설정 (모델, 데이터, 파라미터)
├── src/
│   ├── data/
│   │   ├── triviaqa_loader.py   # TriviaQA rc 데이터 로딩 및 evidence_present 필터링
│   │   └── sampler.py           # NER 태그별 층화 추출
│   ├── probes/
│   │   ├── knowledge_probe.py   # Exp. 2-0: Knowledge Probe 실행
│   │   └── answer_matcher.py    # Gold answer 매칭 (alias + LLM fallback)
│   ├── prompts/
│   │   ├── ner_prompt.py        # NER 프롬프트 (Lee et al. Figure 6 기반)
│   │   └── probe_prompts.py     # Knowledge Probe 프롬프트 (방식 A/B)
│   ├── api/
│   │   └── openai_client.py     # OpenAI API 클라이언트 (rate limiting, retry)
│   └── utils/
│       └── logger.py            # 로깅 유틸리티
├── scripts/
│   ├── 00_prepare_data.py       # TriviaQA 다운로드 및 evidence_present 필터링
│   ├── 01_ner_tagging.py        # NER 태깅 + 층화 추출
│   ├── 02_knowledge_probe.py    # Knowledge Probe 실행 (방식 A/B × 3회)
│   └── 03_analyze_probe.py      # 4단계 분류 + 분석 리포트
├── data/                        # 데이터 저장 (git-ignored)
├── results/                     # 실험 결과 저장
├── requirements.txt
├── .env.example
└── .gitignore
```

## 환경 설정

### 1. Python 가상환경 생성

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 4. TriviaQA 데이터 준비

```bash
python scripts/00_prepare_data.py
```

## 실행 순서

```bash
# Step 0: 데이터 준비 (TriviaQA rc 다운로드 + evidence_present 필터링)
python scripts/00_prepare_data.py

# Step 1: NER 태깅 + 층화 추출 (~280-300 문항)
python scripts/01_ner_tagging.py

# Step 2: Knowledge Probe 실행 (방식 A/B × 3회 = ~1,728 API 호출)
python scripts/02_knowledge_probe.py

# Step 3: 4단계 분류 분석 + 리포트 생성
python scripts/03_analyze_probe.py
```

## 실험 설계 (Exp. 2-0: Knowledge Probe)

Judge에게 context 없이 질문만 제시하여 사전지식 보유 여부를 확인합니다.

| 방식 | 프롬프트 | 특성 |
|------|----------|------|
| A (Bare question) | 질문만 제시 | knows rate 과소 추정 가능 |
| B (Knowledge-eliciting) | "내부 지식으로만 답하라" 명시 | knows rate 과대 추정 가능 |

### 4단계 분류

| 정답 횟수 | 분류 | 해석 |
|-----------|------|------|
| 3/3 | strong-knows | 확실하게 인코딩된 지식 |
| 2/3 | weak-knows | 불확실하지만 접근 가능 |
| 1/3 | guess | 우연히 맞춘 가능성 |
| 0/3 | doesn't-know | 해당 지식 미보유 |

## References

- Lee et al. (2026). *Judging Against the Reference*. arXiv:2601.07506.
- Gekhman et al. (2026). *Thinking to Recall*. arXiv:2603.09906.

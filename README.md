# LLM-as-Judge PKI (Parametric Knowledge Interference) Study

LLM-as-a-Judge가 context-grounded QA 평가에서 제공된 context 대신 자신의
**parametric knowledge에 의존해 판정을 왜곡하는 현상(PKI)** 을 정량화하는 연구 코드베이스.
Lee et al. (2026)의 swapped-reference 프레임워크를 context 차원으로 확장한다.

## 현재 진행 상태 (2026-04 기준)

| 실험 | 상태 | 핵심 결과 |
|------|------|-----------|
| Exp.1 — 3-Prompt Comparison | 완료 | P-Lee-Standard 채택 (Evidence-absent PKI 22.2%, 최저) |
| Exp.2-0 — Knowledge Probe | 완료 | strong-knows 90.6% (Method A), Method A·B 일치도 κ=0.693 |
| Exp.2-1 — Baseline (원본 context) | 완료 | ACC_orig 99.4% (994/1000) |
| Exp.2-2 — Context-Swap | 완료 | Same PKI 4.81% / Cross PKI 4.70% (H-PKI-2 기각) |
| **Exp.2-3 — Prompt Mitigation** | **다음 작업** | P-Lee-Standard vs Direct vs CoT |
| Exp.3 — Model Expansion | 예정 | GPT-4o, Claude 3.5 Haiku |

가설 검증 현황: H-PKI-1 지지 / H-PKI-2 기각 / H-KNOW 강력 지지 / H-ENT-1·2 지지.

## 핵심 지표: PKI + CAR + ORR = 1

Swap 조건(질문과 무관한 context를 제시)에서 judge의 응답 분포를 세 부분으로 분해한다.

| 지표 | 정의 | 의미 |
|------|------|------|
| **PKI Rate** | swap context에서도 CORRECT 판정 비율 | parametric knowledge 누설 |
| **CAR** (Context Adherence) | NOT_ATTEMPTED로 충실 판정한 비율 | context를 따르는 정상 행동 |
| **ORR** (Over-Rejection) | INCORRECT로 과잉 거부한 비율 | 무관 context에 과민 반응 |

세 지표의 합은 항상 1이며, 이상적 judge는 PKI=0, CAR=1, ORR=0.

## 프로젝트 구조

```
llm-judge-pki/
├── configs/
│   └── config.yaml                 # 실험 설정 (모델, 데이터, 파라미터)
├── src/
│   ├── data/
│   │   ├── triviaqa_loader.py      # TriviaQA rc 로딩 + evidence_present 필터
│   │   ├── sampler.py              # NER 태그별 층화 추출
│   │   ├── context_swap.py         # Same-Type / Cross-Type swap 페어 생성
│   │   └── nq_loader.py            # NQ 시도 (현재 미사용, 보존)
│   ├── prompts/
│   │   ├── ner_prompt.py           # NER 프롬프트 (Lee et al. Figure 6)
│   │   ├── probe_prompts.py        # Knowledge Probe 프롬프트 (Method A/B)
│   │   └── judge_prompts.py        # P-Lee-Standard judge 프롬프트
│   ├── probes/
│   │   ├── knowledge_probe.py      # Exp.2-0 Knowledge Probe 실행
│   │   └── answer_matcher.py       # Gold answer 매칭 (alias + LLM fallback)
│   ├── evaluation/
│   │   └── judge_runner.py         # PKI/CAR/ORR 계산, 30회 반복 집계
│   ├── api/
│   │   └── openai_client.py        # OpenAI Batch API submit-poll-collect
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── 00_prepare_data.py          # TriviaQA 다운로드 + evidence_present 필터
│   ├── 01_ner_tagging.py           # NER 태깅 + 층화 추출
│   ├── 01a_evidence_curation.py    # Curated Evidence-Centric Subset 생성 (1,000)
│   ├── 02_knowledge_probe.py       # Exp.2-0 Knowledge Probe (Method A/B × 3회)
│   ├── 03_analyze_probe.py         # Probe 4단계 분류 + 분석 리포트
│   ├── 04_baseline.py              # Exp.2-1 원본 context 평가 (×30회)
│   └── 05_context_swap.py          # Exp.2-2 Same/Cross swap 평가 (×30회)
├── results/
│   ├── probe/                      # Exp.2-0 산출물
│   ├── baseline/                   # Exp.2-1 산출물
│   └── context_swap/               # Exp.2-2 산출물
├── data/                           # raw/processed (git-ignored)
├── logs/                           # 실험 로그
├── tests/
├── requirements.txt
├── .env.example
└── .gitignore
```

## 환경 설정

### 1. Python 가상환경

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

핵심 라이브러리: `openai>=1.30.0`, `pandas`, `numpy`, `datasets`, `tenacity`,
`scipy`, `python-dotenv`, `tqdm`, `tabulate`.

### 3. 환경 변수

```bash
cp .env.example .env
# OPENAI_API_KEY 입력 (Exp.3부터는 ANTHROPIC_API_KEY 추가)
```

## 운영 사양

| 항목 | 값 |
|------|-----|
| Judge 모델 | `gpt-4o-mini-2024-07-18` |
| NER 태깅 모델 | `gpt-4o-2024-08-06` |
| Sampling | T=0, seed=42 |
| 반복 횟수 (Exp.2-1, 2-2) | 30회 / 아이템 |
| 반복 횟수 (Exp.2-0 Probe) | 3회 / 아이템 / 방법 |
| API 호출 패턴 | OpenAI Batch API (submit → poll → collect) |

비용 절감을 위해 동기 호출(`client.chat.completions.create`)이 아니라
Batch API로 JSONL 요청을 일괄 제출한다. 신규 실험도 동일 패턴 유지.

## 실행 순서

```bash
# Step 0: TriviaQA rc 다운로드 + evidence_present 필터
python scripts/00_prepare_data.py

# Step 1: NER 태깅 (GPT-4o 기반, spaCy 미사용)
python scripts/01_ner_tagging.py

# Step 1a: Curated Evidence-Centric Subset 1,000 아이템 생성
python scripts/01a_evidence_curation.py

# Step 2: Exp.2-0 Knowledge Probe (Method A/B × 3회 = 6,000 호출)
python scripts/02_knowledge_probe.py

# Step 3: Probe 4단계 분류 + 분석 리포트
python scripts/03_analyze_probe.py

# Step 4: Exp.2-1 Baseline — 원본 context 30회 평가
python scripts/04_baseline.py

# Step 5: Exp.2-2 Context-Swap — Same/Cross 각 30회 평가
python scripts/05_context_swap.py
```

## 실험 설계 및 결과

### 데이터 큐레이션

TriviaQA rc subset에서 evidence_present 필터(Stage 1) →
Judge-as-Verifier 검증(Stage 2, P-Lee-Standard) → NER 태깅 +
랜덤 샘플링(Stage 3)을 거쳐 1,000 아이템의 Curated Evidence-Centric
Subset을 확보. Context-Swap의 유효 N은 swap 파트너 가용성에 따라
조건별로 다르다 (Same 997, Cross 999) — 자세한 분모 차이는
Exp.2-2 표 주석 참조.

### Exp.2-0: Knowledge Probe

Judge에게 context 없이 질문만 제시하여 사전지식 보유 여부를 측정.

| 방식 | 프롬프트 | 채택 |
|------|----------|------|
| Method A (Bare question) | 질문만 제시 | **다운스트림 분석 기준** |
| Method B (Knowledge-eliciting) | "내부 지식으로만 답하라" 명시 | sensitivity check |

**4단계 분류** (3회 self-consistency 기준):

| 정답 횟수 | 분류 | Method A | Method B |
|-----------|------|----------|----------|
| 3/3 | strong-knows | 90.6% | 91.7% |
| 2/3 | weak-knows | 1.7% | 1.6% |
| 1/3 | guess | 2.3% | 1.2% |
| 0/3 | doesn't-know | 5.4% | 5.5% |

Method A·B 일치도 Cohen's κ = 0.693 (substantial agreement).
H-KNOW 등 후속 분석은 **Method A를 ground truth로 사용**, Method B 결과는
appendix에 비교 테이블로 보고.

### Exp.2-1: Baseline (원본 context)

원본 context를 그대로 제공한 조건에서 judge 정확도 측정.

- N=1,000 아이템 × 30회
- ACC_orig = **99.4%** (994/1000), 평균 verdict entropy 0.008
- Judge가 정상 context에서 거의 완벽하게 작동함을 확인 → swap 조건의 PKI는
  judge 능력 부족이 아니라 parametric knowledge 누설로 해석 가능

### Exp.2-2: Context-Swap

질문과 무관한 context를 swap으로 제시. PKI/CAR/ORR 분해.

| 조건 | N | PKI | CAR | ORR |
|------|----|------|------|------|
| Same-Type swap (동일 NER 타입 내) | 997 | **4.81%** | 9.23% | 85.96% |
| Cross-Type swap (타 NER 타입) | 999 | **4.70%** | 15.32% | 79.98% |

> Curated Subset 1,000 중 큐레이션 단계에서 1건 제외(→ Cross 999), Same은
> 동일 NER 태그 내 swap 파트너가 필요하므로 singleton 태그(LAW, LANGUAGE)
> 2건이 추가 제외되어 유효 N=997.

- Same vs Cross PKI 차이 약 0.1pp → **H-PKI-2 (PKI는 swap 거리에 비례) 기각**
- H-KNOW: PKI는 strong-knows에서만 5%대로 발생, weak/guess/doesn't-know는
  사실상 0% → **지식 보유가 PKI의 필요조건**이라는 강력한 증거

## Scope Out

다음 항목은 현재 연구 범위에서 **제외**됨:

- CoQA 벤치마크 → TriviaQA 단일화
- Medical RAG testbed → 연구 피봇 후 미사용
- Axis 2 preference pair generation → 범위 축소
- CPAG / Semantic Entropy 프레이밍 → PKI/CAR/ORR로 대체
- spaCy NER (`en_core_web_sm`) → GPT-4o 기반 NER로 교체
- P-Thakur, P-CLEV 프롬프트 → Exp.1 비교용으로만 사용, Exp.2 이후 P-Lee 계열만

## References

- Lee, D., Hwang, Y., Kang, T., Lee, M., Chae, Y., & Jung, K. (2026).
  *Judging Against the Reference: Uncovering Knowledge-Driven Failures
  in LLM-Judges on QA Evaluation*. arXiv:2601.07506.
- Gekhman, Z., Aharoni, R., Ofek, E., Geva, M., Reichart, R., & Herzig, J. (2026).
  *Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs*.
  arXiv:2603.09906.

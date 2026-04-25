# CLAUDE.md

이 파일은 Claude Code가 매 세션 시작 시 자동으로 로드하는 컨텍스트 파일이다.
연구 배경 전반은 claude.ai의 Project Instructions에 있으며, 이 파일은 **코드 작업에 특화된 정보**만 다룬다.

---

## 프로젝트 한 줄 요약

LLM-as-a-Judge의 **Parametric Knowledge Interference (PKI)** 를 측정하는 연구. Lee et al.(2026)의 swapped-reference 프레임워크를 context 차원으로 확장하여, judge가 제공된 context 대신 자신의 parametric knowledge에 의존해 평가하는 현상을 정량화한다.

핵심 메트릭: **PKI + CAR + ORR = 1**
- PKI Rate: swap context에서도 CORRECT 판정한 비율
- CAR (Context Adherence): NOT_ATTEMPTED로 충실 판정한 비율
- ORR (Over-Rejection): INCORRECT로 과잉 거부한 비율

---

## 현재 진행 상태

| 실험 | 상태 | 핵심 결과 |
|------|------|-----------|
| Exp.1 (3-Prompt Comparison) | 완료 | P-Lee-Standard 채택 (Evidence-absent PKI 22.2%, 최저) |
| Exp.2-0 (Knowledge Probe) | 완료 | strong-knows 90.6%, κ=0.693 |
| Exp.2-1 (Baseline) | 완료 | ACC_orig 99.4% |
| Exp.2-2 (Context-Swap) | 완료 | Same PKI 4.81% (N=997), Cross PKI 4.70% (N=999) |
| Exp.2-3 (Prompt Mitigation) | 완료 | Direct PKI 0.70%, CoT PKI 1.10% (vs Standard 4.81%) |
| **Exp.3 (Model Expansion)** | **다음 작업** | GPT-4o, Claude 3.5 Haiku |

주요 가설: H-PKI-1 지지, H-PKI-2 기각, H-KNOW 강력 지지(prompt-invariant — Exp.2-3에서 모든 prompt가 strong-knows에서만 PKI 발생), H-ENT-1/2 지지. Direct·CoT 모두 PKI 유의 감소; CoT의 reasoning trace는 PKI 자기-정당화 통로로 작용 (Exp.2-3 정성 분석).

> Context-Swap 유효 N 비대칭: Curated Subset 1,000 중 큐레이션 단계에서 1건 제외(→ Cross 999), Same은 동일 NER 태그 내 swap 파트너가 필요하므로 singleton 태그(LAW, LANGUAGE) 2건이 추가 제외되어 N=997.

---

## 환경 셋업

### 가상환경 활성화

```bash
source .venv/bin/activate
```

세션 시작 시 항상 먼저 활성화한다. `.venv/`는 git에서 제외되어 있다.

### 의존성 설치

```bash
pip install -r requirements.txt
```

핵심 라이브러리: `openai>=1.30.0`, `pandas`, `numpy`, `datasets` (TriviaQA), `tenacity` (재시도), `scipy` (가설 검정), `python-dotenv`, `tqdm`, `tabulate`.

### 환경변수

`.env` 파일에서 로드 (절대 커밋 금지, `.gitignore`에 등록됨):

- `OPENAI_API_KEY`: 필수 (Judge 모델, NER 태깅, Batch API)
- `ANTHROPIC_API_KEY`: Exp.3 (Claude 3.5 Haiku) 시점부터 필요

신규 셋업 시 `.env.example` 복사 후 키 입력.

---

## 디렉토리 가이드

```
llm-judge-pki/
├── configs/                # 실험별 YAML 설정
├── data/
│   ├── raw/                # TriviaQA 원본 (gitignored)
│   ├── processed/          # 큐레이션 1,000 아이템, NER 태깅 (gitignored)
│   └── batch_outputs/      # OpenAI Batch API 응답 raw
├── logs/                   # 실행 로그
├── results/
│   ├── probe/              # Exp.2-0 Knowledge Probe
│   ├── baseline/           # Exp.2-1 원본 context 평가
│   ├── context_swap/       # Exp.2-2 Same/Cross swap
│   └── prompt_mitigation/  # Exp.2-3 Direct/CoT (Standard는 context_swap/same 재사용)
├── scripts/                # 일회성 분석 스크립트
├── src/
│   ├── api/                # OpenAI Batch API submit/collect wrapper
│   ├── data/               # TriviaQA 큐레이션, NER 태깅 파이프라인
│   ├── prompts/            # P-Lee-Standard/Direct/CoT 정의
│   ├── probes/             # Knowledge Probe (self-consistency)
│   ├── evaluation/         # PKI/CAR/ORR 계산, 가설 검정
│   └── utils/              # 공통 유틸
└── tests/
```

`.gitignore`로 인해 `data/*.json`, `data/*.jsonl`, `data/*.parquet`, `data/raw/`는 추적되지 않는다. 새 데이터 산출물은 이 패턴에 맞춰 저장한다.

---

## 코드 컨벤션

- **언어**: Python 3.12
- **주석 / docstring**: 영어
- **대화 (사용자와의 응답)**: 한국어
- **결과 파일**: JSONL 형식, `results/{실험명}/` 하위
- **로깅**: `logs/`에 실험별 로그 파일
- **분석 스크립트**: 가설별 (H-PKI-1/2, H-KNOW 등) 검정 결과를 명시적으로 출력

### Batch API 패턴 (중요)

비용 절감을 위해 **OpenAI Batch API** 를 사용한다. 동기 호출 (`client.chat.completions.create`)이 아니라 **submit-poll-collect** 패턴:

1. `src/api/`에서 요청 JSONL 작성 → batch 제출
2. 폴링 또는 완료 알림 후 결과 다운로드
3. `data/batch_outputs/`에 raw 응답 저장
4. 파싱 후 `results/{실험명}/`로 정제

새 실험 추가 시 이 패턴을 그대로 따른다. 단발 디버깅 외에는 동기 호출 금지.

### 모델 설정

- **Judge 모델**: `gpt-4o-mini-2024-07-18` (T=0, seed=42)
- **NER 태깅**: `gpt-4o-2024-08-06`
- **반복 횟수**: 30회/아이템 (Exp.2 기준)

---

## 자주 쓰는 명령

```bash
# 가상환경 + 의존성
source .venv/bin/activate
pip install -r requirements.txt

# 테스트
pytest tests/

# 결과 디렉토리 확인
ls -la results/{probe,baseline,context_swap}/

# 배치 응답 raw 확인
ls -la data/batch_outputs/
```

---

## 주의사항 / Scope Out

다음 항목은 **현재 연구 범위에서 제외**되었다. 관련 코드를 새로 작성하지 말 것:

- **CoQA 벤치마크** → TriviaQA 단일화
- **Medical RAG testbed** → 연구 피봇 후 미사용
- **Axis 2 preference pair generation** → 범위 축소
- **CPAG 메트릭** → PKI/CAR/ORR로 대체
- **Semantic Entropy 프레이밍** → PKI 측정 프레임워크로 진화
- **spaCy NER 태깅** → Exp.2부터 GPT-4o 기반 NER로 교체. `en_core_web_sm` 신규 사용 금지
- **P-Thakur, P-CLEV 프롬프트** → Exp.1 비교용으로만 사용. Exp.2 이후 P-Lee 계열만

### 보안

- `.env`는 절대 커밋 금지. `.gitignore`에 등록되어 있으나 새 키 추가 시 재확인.
- API 키를 코드 본문에 하드코딩 금지. 항상 `os.getenv("...")` 또는 `python-dotenv`로 로드.

---

## 다음 작업 (Exp.3 Model Expansion 준비)

**목표**: gpt-4o-mini 외 다른 judge 모델에서 Exp.2-2/2-3의 PKI 패턴이 재현되는지 검증.

**조건**:
- Exp.2-2/2-3과 동일 데이터 (Same-Type swap, N=997, 30회 판정/아이템)
- 모델: GPT-4o (`gpt-4o-2024-08-06`), Claude 3.5 Haiku (`claude-3-5-haiku-20241022`)
- 프롬프트: P-Lee-Standard 우선, Direct/CoT는 ROI 검토 후

**구현 시 점검**:
1. `src/api/`에 Anthropic 클라이언트 추가 (`anthropic` SDK), `ANTHROPIC_API_KEY` 셋업
2. `src/evaluation/judge_runner.py`를 provider 추상화 — OpenAI / Anthropic 양쪽 호출 인터페이스 통합
3. Anthropic Message Batches API submit-poll-collect 패턴 (OpenAI Batch API와 별개)
4. 결과 저장 위치: `results/model_expansion/{model}/`
5. parse_verdict가 Anthropic 응답 형식(content blocks)에도 호환되는지 확인

**예상 호출 규모**: 모델 2 × 997 × 30 = 약 60,000 (Standard만 기준; Direct/CoT 추가 시 ×3)

---

## 참고

- **GitHub**: `jwrhee007/llm-judge-pki` (private)
- **Reference paper**: Lee et al. (2026), arXiv:2601.07506
- **Backbone prompt**: Lee et al. Table 9/10/11 (P-Lee-Standard)
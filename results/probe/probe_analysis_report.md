# Knowledge Probe Analysis Report (Exp. 2-0)

## 1. Overview

- Total items: **250**
- Method A (Bare question): 268 items
- Method B (Knowledge-eliciting): 268 items

## 2. Overall 4-Level Classification

| Level | Method A | A (%) | Method B | B (%) |
|-------|----------|-------|----------|-------|
| strong-knows | 238 | 88.8% | 232 | 86.6% |
| weak-knows | 3 | 1.1% | 3 | 1.1% |
| guess | 2 | 0.7% | 4 | 1.5% |
| doesn't-know | 25 | 9.3% | 29 | 10.8% |

## 3. Method A vs B Agreement

- **Exact agreement rate**: 96.4%
- **Cohen's Kappa**: 0.828

- **Interpretation**: Almost perfect agreement

## 4. Cross-tabulation (Method A × Method B)

| level_A      |   strong-knows |   weak-knows |   guess |   doesn't-know |   Total |
|:-------------|---------------:|-------------:|--------:|---------------:|--------:|
| strong-knows |            218 |            1 |       2 |              2 |     223 |
| weak-knows   |              1 |            1 |       1 |              0 |       3 |
| guess        |              0 |            0 |       0 |              2 |       2 |
| doesn't-know |              0 |            0 |       0 |             22 |      22 |
| Total        |            219 |            2 |       3 |             26 |     250 |

## 5. Disagreement Analysis

Total disagreements: 9

| Method A → Method B | Count |
|---------------------|-------|
| strong-knows → doesn't-know | 2 |
| guess → doesn't-know | 2 |
| strong-knows → guess | 2 |
| weak-knows → guess | 1 |
| strong-knows → weak-knows | 1 |
| weak-knows → strong-knows | 1 |

## 6. NER Tag Breakdown

| ner_tag     |   n |   A_strong-knows |   A_weak-knows |   A_guess |   A_doesn't-know |   B_strong-knows |   B_weak-knows |   B_guess |   B_doesn't-know |
|:------------|----:|-----------------:|---------------:|----------:|-----------------:|-----------------:|---------------:|----------:|-----------------:|
| CARDINAL    |  20 |               19 |              0 |         1 |                0 |               18 |              0 |         1 |                1 |
| DATE        |  20 |               17 |              0 |         0 |                3 |               17 |              0 |         0 |                3 |
| EVENT       |  20 |               20 |              0 |         0 |                0 |               20 |              0 |         0 |                0 |
| FAC         |  18 |               15 |              0 |         0 |                3 |               15 |              0 |         0 |                3 |
| GPE         |  20 |               18 |              0 |         0 |                2 |               18 |              0 |         0 |                2 |
| LANGUAGE    |  15 |               13 |              1 |         0 |                1 |               14 |              0 |         0 |                1 |
| LOC         |  20 |               19 |              0 |         0 |                1 |               19 |              0 |         0 |                1 |
| NORP        |  19 |               15 |              1 |         1 |                2 |               15 |              1 |         0 |                3 |
| ORDINAL     |   6 |                6 |              0 |         0 |                0 |                6 |              0 |         0 |                0 |
| ORG         |  20 |               19 |              0 |         0 |                1 |               19 |              0 |         0 |                1 |
| PERSON      |  20 |               18 |              0 |         0 |                2 |               18 |              0 |         0 |                2 |
| PRODUCT     |  20 |               18 |              1 |         0 |                1 |               16 |              0 |         1 |                3 |
| QUANTITY    |  12 |                9 |              0 |         0 |                3 |                7 |              1 |         1 |                3 |
| WORK_OF_ART |  20 |               17 |              0 |         0 |                3 |               17 |              0 |         0 |                3 |

## 7. Knows Rate Comparison

*Knows rate = (strong-knows + weak-knows) / total*

- Method A knows rate: **89.9%** (241/268)
- Method B knows rate: **87.7%** (235/268)

Method A는 Method B보다 knows rate이 2.2pp 높다.

## 8. Next Steps

- 방식 채택 기준: Exp. 2-2 Context-Swap 실행 후 PKI rate와의 상관이 더 높은 방식을 최종 채택
- 채택하지 않은 방식의 결과는 appendix에 비교 테이블로 보고
- H-KNOW 가설 검증: PKI rate가 knowledge probe 단계에 따라 단조 증가하는지 확인 (strong-knows > weak-knows > guess > doesn't-know)

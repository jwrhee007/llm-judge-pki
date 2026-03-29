# Knowledge Probe Analysis Report (Exp. 2-0)

## 1. Overview

- Total items: **944**
- Method A (Bare question): 1000 items
- Method B (Knowledge-eliciting): 1000 items

## 2. Overall 4-Level Classification

| Level | Method A | A (%) | Method B | B (%) |
|-------|----------|-------|----------|-------|
| strong-knows | 906 | 90.6% | 917 | 91.7% |
| weak-knows | 17 | 1.7% | 16 | 1.6% |
| guess | 23 | 2.3% | 12 | 1.2% |
| doesn't-know | 54 | 5.4% | 55 | 5.5% |

## 3. Method A vs B Agreement

- **Exact agreement rate**: 94.9%
- **Cohen's Kappa**: 0.693

- **Interpretation**: Substantial agreement

## 4. Cross-tabulation (Method A × Method B)

| level_A      |   strong-knows |   weak-knows |   guess |   doesn't-know |   Total |
|:-------------|---------------:|-------------:|--------:|---------------:|--------:|
| strong-knows |            846 |            7 |       0 |              2 |     855 |
| weak-knows   |              8 |            4 |       2 |              2 |      16 |
| guess        |              8 |            3 |       6 |              6 |      23 |
| doesn't-know |              4 |            2 |       4 |             40 |      50 |
| Total        |            866 |           16 |      12 |             50 |     944 |

## 5. Disagreement Analysis

Total disagreements: 48

| Method A → Method B | Count |
|---------------------|-------|
| guess → strong-knows | 8 |
| weak-knows → strong-knows | 8 |
| strong-knows → weak-knows | 7 |
| guess → doesn't-know | 6 |
| doesn't-know → strong-knows | 4 |
| doesn't-know → guess | 4 |
| guess → weak-knows | 3 |
| weak-knows → guess | 2 |
| doesn't-know → weak-knows | 2 |
| weak-knows → doesn't-know | 2 |

## 6. NER Tag Breakdown

| ner_tag     |   n |   A_strong-knows |   A_weak-knows |   A_guess |   A_doesn't-know |   B_strong-knows |   B_weak-knows |   B_guess |   B_doesn't-know |
|:------------|----:|-----------------:|---------------:|----------:|-----------------:|-----------------:|---------------:|----------:|-----------------:|
| CARDINAL    |  18 |               16 |              1 |         0 |                1 |               16 |              0 |         1 |                1 |
| DATE        |  16 |               14 |              0 |         0 |                2 |               14 |              0 |         0 |                2 |
| EVENT       |  22 |               21 |              0 |         0 |                1 |               22 |              0 |         0 |                0 |
| FAC         |  18 |               16 |              1 |         0 |                1 |               17 |              0 |         1 |                0 |
| GPE         | 198 |              192 |              1 |         0 |                5 |              190 |              2 |         0 |                6 |
| LANGUAGE    |   2 |                2 |              0 |         0 |                0 |                2 |              0 |         0 |                0 |
| LAW         |   1 |                1 |              0 |         0 |                0 |                1 |              0 |         0 |                0 |
| LOC         |  47 |               41 |              2 |         0 |                4 |               40 |              2 |         0 |                5 |
| NORP        |   9 |                9 |              0 |         0 |                0 |                9 |              0 |         0 |                0 |
| ORDINAL     |   2 |                2 |              0 |         0 |                0 |                2 |              0 |         0 |                0 |
| ORG         |  84 |               80 |              2 |         1 |                1 |               79 |              2 |         1 |                2 |
| PERSON      | 338 |              288 |              9 |        14 |               27 |              298 |              7 |         8 |               25 |
| PRODUCT     |  80 |               70 |              0 |         7 |                3 |               74 |              2 |         0 |                4 |
| QUANTITY    |   2 |                2 |              0 |         0 |                0 |                2 |              0 |         0 |                0 |
| WORK_OF_ART | 107 |              101 |              0 |         1 |                5 |              100 |              1 |         1 |                5 |

## 7. Knows Rate Comparison

*Knows rate = (strong-knows + weak-knows) / total*

- Method A knows rate: **92.3%** (923/1000)
- Method B knows rate: **93.3%** (933/1000)

Method B는 Method A보다 knows rate이 1.0pp 높다. Knowledge-eliciting prompt가 모델의 지식 recall을 더 잘 유도함.

## 8. Next Steps

- 방식 채택 기준: Exp. 2-2 Context-Swap 실행 후 PKI rate와의 상관이 더 높은 방식을 최종 채택
- 채택하지 않은 방식의 결과는 appendix에 비교 테이블로 보고
- H-KNOW 가설 검증: PKI rate가 knowledge probe 단계에 따라 단조 증가하는지 확인 (strong-knows > weak-knows > guess > doesn't-know)

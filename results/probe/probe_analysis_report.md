# Knowledge Probe Analysis Report (Exp. 2-0)

## 1. Overview

- Total items: **956**
- Method A (Bare question): 1000 items
- Method B (Knowledge-eliciting): 1000 items

## 2. Overall 4-Level Classification

| Level | Method A | A (%) | Method B | B (%) |
|-------|----------|-------|----------|-------|
| strong-knows | 888 | 88.8% | 882 | 88.2% |
| weak-knows | 18 | 1.8% | 21 | 2.1% |
| guess | 22 | 2.2% | 18 | 1.8% |
| doesn't-know | 72 | 7.2% | 79 | 7.9% |

## 3. Method A vs B Agreement

- **Exact agreement rate**: 93.9%
- **Cohen's Kappa**: 0.712

- **Interpretation**: Substantial agreement

## 4. Cross-tabulation (Method A × Method B)

| level_A      |   strong-knows |   weak-knows |   guess |   doesn't-know |   Total |
|:-------------|---------------:|-------------:|--------:|---------------:|--------:|
| strong-knows |            831 |           10 |       4 |              4 |     849 |
| weak-knows   |              7 |            2 |       5 |              3 |      17 |
| guess        |              3 |            5 |       5 |              8 |      21 |
| doesn't-know |              2 |            3 |       4 |             60 |      69 |
| Total        |            843 |           20 |      18 |             75 |     956 |

## 5. Disagreement Analysis

Total disagreements: 58

| Method A → Method B | Count |
|---------------------|-------|
| strong-knows → weak-knows | 10 |
| guess → doesn't-know | 8 |
| weak-knows → strong-knows | 7 |
| weak-knows → guess | 5 |
| guess → weak-knows | 5 |
| strong-knows → doesn't-know | 4 |
| doesn't-know → guess | 4 |
| strong-knows → guess | 4 |
| doesn't-know → weak-knows | 3 |
| weak-knows → doesn't-know | 3 |

## 6. NER Tag Breakdown

| ner_tag     |   n |   A_strong-knows |   A_weak-knows |   A_guess |   A_doesn't-know |   B_strong-knows |   B_weak-knows |   B_guess |   B_doesn't-know |
|:------------|----:|-----------------:|---------------:|----------:|-----------------:|-----------------:|---------------:|----------:|-----------------:|
| CARDINAL    |  34 |               29 |              4 |         1 |                0 |               30 |              0 |         2 |                2 |
| DATE        |  31 |               29 |              1 |         0 |                1 |               29 |              0 |         0 |                2 |
| EVENT       |  22 |               21 |              1 |         0 |                0 |               22 |              0 |         0 |                0 |
| FAC         |  22 |               20 |              1 |         0 |                1 |               20 |              0 |         2 |                0 |
| GPE         | 214 |              194 |              2 |         2 |               16 |              191 |              6 |         0 |               17 |
| LANGUAGE    |   5 |                5 |              0 |         0 |                0 |                5 |              0 |         0 |                0 |
| LOC         |  48 |               44 |              0 |         2 |                2 |               44 |              1 |         1 |                2 |
| MONEY       |   1 |                1 |              0 |         0 |                0 |                1 |              0 |         0 |                0 |
| NORP        |   9 |                8 |              0 |         0 |                1 |                8 |              0 |         0 |                1 |
| ORDINAL     |   1 |                1 |              0 |         0 |                0 |                1 |              0 |         0 |                0 |
| ORG         |  80 |               68 |              2 |         2 |                8 |               67 |              1 |         2 |               10 |
| PERSON      | 300 |              266 |              3 |         9 |               22 |              261 |             10 |         7 |               22 |
| PRODUCT     |  76 |               67 |              1 |         2 |                6 |               67 |              0 |         2 |                7 |
| QUANTITY    |   1 |                0 |              0 |         0 |                1 |                0 |              0 |         0 |                1 |
| WORK_OF_ART | 112 |               96 |              2 |         3 |               11 |               97 |              2 |         2 |               11 |

## 7. Knows Rate Comparison

*Knows rate = (strong-knows + weak-knows) / total*

- Method A knows rate: **90.6%** (906/1000)
- Method B knows rate: **90.3%** (903/1000)

Method A는 Method B보다 knows rate이 0.3pp 높다.

## 8. Next Steps

- 방식 채택 기준: Exp. 2-2 Context-Swap 실행 후 PKI rate와의 상관이 더 높은 방식을 최종 채택
- 채택하지 않은 방식의 결과는 appendix에 비교 테이블로 보고
- H-KNOW 가설 검증: PKI rate가 knowledge probe 단계에 따라 단조 증가하는지 확인 (strong-knows > weak-knows > guess > doesn't-know)

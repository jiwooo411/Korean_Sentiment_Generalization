# 01 Overview

sklearn-only 제약 하에서 한국어 이진 감성 분류를 수행하고 MCC로 평가하는 경쟁 과제의 전체 맥락 정리.

---

## 1.  문제 정의

**태스크**: 한국어 영화 리뷰 → `POSITIVE` / `NEGATIVE` 이진 분류

- 입력: 자연어 텍스트 (한국어, 비정형)
- 출력: `POSITIVE` 또는 `NEGATIVE` 문자열 라벨
- 평가 지표: **MCC (Matthews Correlation Coefficient)**
- 제출 방식: 예측 CSV → Streamlit 웹앱 업로드 (리더보드 실시간 반영)

> 출처: `중간고사_guide.md §1`

---

## 2.  데이터

| 구분 | 파일 | 행 수 | 컬럼 | 비고 |
|---|---|---|---|---|
| 학습 | `public_train.csv` | 149,995 | `row_id`, `text`, `label` | 라벨 50:50 균형 |
| 테스트 | `public_test.csv` | 49,997 | `row_id`, `text` | 라벨 없음 |
| 샘플 제출 | `sample_submission.csv` | 49,997 | `row_id`, `pred_label` | 제출 형식 기준 |

- **라벨 분포**: `POSITIVE` / `NEGATIVE` 약 50:50 균형 → 클래스 불균형 문제 없음
- **결측값**: 0건 (별도 처리 불필요)
- **데이터 출처**: NSMC(Naver Sentiment Movie Corpus) 계열

> 출처: `중간고사_guide.md §1` / `OPUS_DESIGN.md §0.D`

---

## 3.  가이드 제약

### 허용 도구

| 범주 | 허용 항목 |
|---|---|
| 분류 알고리즘 | `sklearn`의 모든 분류기 (LogReg, SVM, NaiveBayes, RandomForest, GBM, AdaBoost 등) |
| 텍스트 벡터화 | `sklearn.feature_extraction.text` — TfidfVectorizer, CountVectorizer 등 |
| 전처리/유틸 | pandas, numpy, scipy, re, konlpy, pecab |
| 시각화 | matplotlib, seaborn, plotly |

### 금지 도구

| 범주 | 금지 항목 |
|---|---|
| 외부 ML | XGBoost, LightGBM, CatBoost |
| 딥러닝 | PyTorch, TensorFlow, Keras |
| 사전학습 모델 | HuggingFace Transformers, Word2Vec, FastText (gensim) |
| 자동화 | AutoML (auto-sklearn, FLAML 등) |

> **위반 시 즉시 0점 처리** (`중간고사_guide.md §2`, §11)

---

## 4.  평가 Metric

### MCC (Matthews Correlation Coefficient)

$$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

- **범위**: [-1, 1] — 1(완벽), 0(랜덤), -1(완전 반대)
- **Φ-coefficient와 동일**: 이진 혼동 행렬의 상관계수. 클래스 균형·불균형 모두 robust.
- **Accuracy 대비 장점**: TP·TN·FP·FN 4개 값을 모두 반영 → 예측 편향 감지 가능.
- **50:50 균형 데이터에서도** accuracy보다 strict한 지표.

### Min(Group1, Group2) 운영 방식

```
LB 점수 (전체 49,997행 MCC)  ≠  최종 성적 점수
```

대회 종료 후 교수님이 테스트 데이터를 두 그룹으로 분할해 각각 MCC를 계산하고 **Min(MCC₁, MCC₂)** 를 최종 점수로 결정.

- **함의**: 특정 패턴 과적합 시 한쪽 그룹 점수 하락 → 최종 불리.
- **전략**: LB 단독 최대화보다 두 그룹 모두에서 robust한 일반화 성능 우선.

> 출처: `중간고사_guide.md §9`

---

## 5.  핵심 도전

### 5.1 Sparse-linear 천장 (~0.77)

- TF-IDF + LogReg ElasticNet 계열의 이론적 상한: **MCC ≈ 0.77 전후**
- Drill-05 최종 LB 0.757은 해당 천장의 97% 수준
- 천장 돌파 경로: dense semantic embedding (Drill-06 참조)

### 5.2 Val 과적합 위험

- Validation split이 단일 holdout(20%)이므로 grid search 반복 시 **정보 누출** 발생.
- Stage 8 (HistGBT val 0.7748 → LB 0.750), Stage 9 (132조합 grid val 0.7745 → LB ≤ 0.753) 가 실증.
- 방어책: **CV-LB gap 부호 모니터링**, threshold tuning을 val이 아닌 OOF로 한정.

> 출처: `OPUS_DESIGN.md §1 Stage 8, Stage 9`

---

## 6.  본 프로젝트 Stance

**좋은 점수 ≠ 좋은 모델**

| 판단 기준 | 우선순위 | 근거 |
|---|---|---|
| CV-LB gap 최소화 | **1순위** | val 과적합 신호 직접 감지 |
| LB MCC 절대값 | 2순위 | ceiling 존재, 추가 이득 한계 |
| Val MCC 절대값 | 3순위 | 단일 split → 과적합 위험 |

- "Val이 높아도 LB가 낮으면 채택 안 함" 원칙 일관 적용 (Stage 8, 9, 10 폐기 근거).
- 실험 목적: 일반화 성능 향상 과정의 **투명한 기록** — 재현 가능 + 방어 가능.

> 출처: `OPUS_DESIGN.md §3 T1` / `중간고사_guide.md §9 Min(Group1, Group2)`

---

## 7.  제출 요건 요약

| 항목 | 규칙 | 출처 |
|---|---|---|
| CSV 형식 | `.csv`, UTF-8, `row_id`+`pred_label`, 49,997행 | `중간고사_guide.md §7` |
| 라벨 문자열 | `POSITIVE` / `NEGATIVE` 대문자 정확히 | `§7` |
| LMS 제출 | `학번_이름.ipynb` + `final_pipeline.pkl` 필수 | `§8` |
| 셀 출력 | 모든 셀 출력 보존 (런타임 > 모두 실행 후 제출) | `§8` |
| random_state | `random_state=42` 전 위치 고정 | `§6`, `§10` |
| 경로 | 상대 경로 또는 변수 (`DATA_DIR`) — 절대 경로 금지 | `§10` |
| 마감 | 2026-04-28 (월) 23:59 KST | `§10` |

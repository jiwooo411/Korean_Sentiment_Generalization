# 02 Methodology — 3-View · 4-Model · Augmented Stacking 아키텍처 및 설계 근거

sklearn-only 제약 하에서 MCC 0.757을 달성한 파이프라인의 구조, 하이퍼파라미터 선정 근거, 검증 전략을 기술한다.

---

## 1. Pipeline Overview

`final/202431626_김지우.ipynb §5.1 (Cell 13)` 기준 전체 데이터 흐름:

```
원문 텍스트 (공개 train 149,995행 / test 49,997행)
        │
        ▼
┌───────────────────────────────────────────┐
│   3-View Preprocessing (§2)              │
│                                           │
│  char_view  ──  보편 정제 원문            │
│  morph_view ──  pecab 형태소 전체         │
│  senti_view ──  KEEP_POS + 불용어 제거    │
└───────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│   Layer 0: 4 Base Models                                        │
│                                                                  │
│  LogReg ElasticNet  ── morph_view + TF-IDF word (1,2)           │
│  LogReg ElasticNet  ── char_view  + TF-IDF char_wb (2,5)        │
│  LogReg ElasticNet  ── senti_view + TF-IDF word (1,2)           │
│  ComplementNB       ── morph_view + TF-IDF word (1,2)           │
└──────────────────────────────────────────────────────────────────┘
        │ 4 × POSITIVE 확률
        ▼
┌─────────────────────────────────────────────────────────┐
│   Layer 0.5: 12 Hand-crafted Meta Features (§4.1)      │
└─────────────────────────────────────────────────────────┘
        │ 4 + 12 = 16차원 입력
        ▼
┌───────────────────────────────────────────────┐
│   Layer 1: Meta-learner                      │
│   StandardScaler → LogReg L2 (C=1.0)        │
│   threshold 0.5 → POSITIVE / NEGATIVE       │
└───────────────────────────────────────────────┘
```

---

## 2. 3-View Preprocessing

### 2.1 설계 근거 — 직교 정보 차원

단일 전처리에 의존하면 그 뷰가 놓치는 신호를 복구할 수 없다.
세 뷰는 **상호 보완적 정보 차원**을 분담하도록 설계하였다.

| View | 처리 방식 | 포착 대상 | TF-IDF analyzer |
|------|-----------|-----------|-----------------|
| `char_view` | URL/HTML 제거 + 반복 정규화 | 표면 패턴 (☆☆☆, !!!, 오타) | `char_wb` |
| `morph_view` | pecab.pos() 형태소 전체 | 문법 구조 (어미/조사) | `word` |
| `sentiment_view` | KEEP_POS 통과 + 불용어 제거 | 의미 집중 핵심어 | `word` |

각 뷰 간 예측 상관은 약 0.5~0.6 수준 → 진정한 다양성 확보 (`final/202431626_김지우.ipynb §7 (Cell 17)` 참고).

### 2.2 Blacklist 정제 원칙

화이트리스트(허용 POS 지정) 대신 **블랙리스트(제거 대상만 열거)** 를 채택한다.

| 제거 대상 | 정규식 | 보존 대상 |
|-----------|--------|-----------|
| URL | `https?://\S+\|www\.\S+` | ☆♥-_- 등 감성 부호 |
| HTML 태그 | `<[^>]+>` | 느낌표, 물음표, 물결 |
| 제어문자 | `[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]+` | ㅋㅋㅋ, ㅠㅜ 등 감성 자모 |
| 4회 이상 반복 | `(.)\1{3,}` → 3회 정규화 | 반복 3회 이하는 원본 유지 |

화이트리스트(예: POS 9종 한정)는 어미(EC/EF/ETM) 정보까지 제거하여 시제·존칭 신호를 손실시킨다 (Appendix A1 참고).

```python
RE_URL     = re.compile(r"https?://\S+|www\.\S+")
RE_HTML    = re.compile(r"<[^>]+>")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]+")
RE_REPEAT  = re.compile(r"(.)\1{3,}")  # 반복 4회+ → 3회로 정규화

def make_char_view(text: str) -> str:
    """View A: URL/HTML/제어문자만 제거, 보편 감성 부호 보존."""
    if not isinstance(text, str): return ""
    t = RE_URL.sub(" ", text)
    t = RE_HTML.sub(" ", t)
    t = RE_CONTROL.sub(" ", t)
    t = RE_REPEAT.sub(lambda m: m.group(1) * 3, t)
    return RE_SPACE.sub(" ", t).strip()
```

### 2.3 pecab Single-Pass 토큰화

`pecab.pos()` 를 **1회만 호출**하여 `morph_view`와 `sentiment_view`를 동시에 생성한다.
2회 호출 대비 형태소 분석 시간 약 50% 절감 (`final/202431626_김지우.ipynb §4.2 (Cell 9)`).

```python
def tokenize_dual(text: str) -> tuple[str, str]:
    """morph_view + sentiment_view 동시 생성 (pecab 단일 호출)."""
    pairs = _PECAB.pos(text)                          # 1회 호출
    morph_tokens, senti_tokens = [], []
    for w, t in pairs:
        if len(w) == 1 and w in _JAMO:
            continue                                   # 단일 자모 제거 (§2.4)
        morph_tokens.append(w)
        if t in KEEP_POS and w not in STOPWORDS:
            senti_tokens.append(w)
    return " ".join(morph_tokens), " ".join(senti_tokens)
```

실제 처리 시간: train 145,980개 고유 텍스트 기준 4,052.7초.

### 2.4 토큰 단계 Jamo Filter

**정규식이 아닌 토큰 단위**에서 단일 자모를 제거한다.

- 제거: 단일 자모 (길이 1이며 `_JAMO` 집합 소속) — 예: 분리된 `ㅂ`, `ㄱ`
- **보존**: 연속 자모 (길이 > 1) — 예: `ㅋㅋ`, `ㅠㅠ`, `ㅎㅎ`

이 방식은 pecab이 `ㅋㅋㅋ`를 개별 `ㅋ`로 분리하더라도 연속된 경우 char_view에서 n-gram으로 포착되도록 설계되어 있다.

```python
_JAMO = frozenset("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
                  "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# 토큰 단계 필터 (형태소 쌍 순회 중 적용)
if len(w) == 1 and w in _JAMO:
    continue  # 단일 자모 노이즈 제거 (ㅋㅋㅋ는 char_view에서 처리)
```

### 2.5 불용어 처리 + 부정 마커 보호

`sentiment_view` 생성 시 표준 불용어 집합에서 **부정·강조 마커를 자동으로 보호**한다.

```python
NEVER_REMOVE = frozenset({
    "안", "못", "않", "없", "아니",          # 부정 마커
    "정말", "진짜", "매우", "너무", "별로",   # 강조 마커
    "좋", "나쁘", "재미있", "재미없", "최고", # 핵심 감성어
})
STOPWORDS = frozenset({
    "것", "거", "수", "때", "데", "건", "걸", "년", "점", "위",
    "나", "내", "너", "저", "그", "이", "우리",
    "되", "하", "있", "보", "들", "주", "오",
    "같", "또", "더", "이미", "벌써", "그냥", "일",
})
# 보호 조건 위반 시 즉시 에러 (Fail-fast)
assert not (NEVER_REMOVE & STOPWORDS), "stopword safety violation"
```

**View별 TF-IDF 입력 파라미터 표:**

| 파라미터 | morph_view | char_view | sentiment_view |
|----------|------------|-----------|----------------|
| `analyzer` | `word` | `char_wb` | `word` |
| `ngram_range` | `(1, 2)` | `(2, 5)` | `(1, 2)` |
| `min_df` | `3` | `5` | `3` |
| `max_df` | `0.95` | `0.95` | `0.95` |
| `sublinear_tf` | `True` | `True` | `True` |
| `max_features` | `300,000` | `300,000` | `300,000` |
| `lowercase` | 기본값 | `False` | 기본값 |
| 실제 피처 수 (80% 학습) | 103,492 | 207,448 | 49,417 |

---

## 3. 4-Model Base Ensemble

### 3.1 모델 구성

| 인덱스 | 모델 | View | Paradigm |
|--------|------|------|----------|
| M0 | LogReg ElasticNet | morph_view | Discriminative |
| M1 | LogReg ElasticNet | char_view | Discriminative |
| M2 | LogReg ElasticNet | sentiment_view | Discriminative |
| M3 | ComplementNB | morph_view | **Generative** |

각 모델의 POSITIVE 확률 4개를 메타 학습기에 전달한다 (Soft voting 방식).

### 3.2 ComplementNB 추가 이유 — Discriminative vs Generative Paradigm 다양성

3-View에 LogReg × 3만 사용하면 **솔버 차이는 있어도 모델 클래스가 동일**하다.
예측 상관이 높아 앙상블 효과가 제한된다.

| 모델 | 분류 패러다임 | 추론 방향 | 노이즈 강건성 |
|------|---------------|-----------|---------------|
| LogReg ElasticNet | Discriminative | P(y\|x) 직접 모델링 | 중간 |
| ComplementNB | **Generative** | P(x\|y) × P(y) → 역추정 | **높음** (Naive 독립 가정) |

ComplementNB는 클래스 불균형에 MultinomialNB보다 robust하며, 짧은 리뷰(평균 35자)에 특화된 텍스트 분류기다.
단독 성능은 낮으나 앙상블에서 LogReg와 다른 오답 패턴을 제공하여 시너지를 발생시킨다.

실증: 3-view LB 0.750 → 4-model LB 0.753 (**+0.003**).

### 3.3 Hyperparameter 표

| 구성요소 | 파라미터 | 값 | 근거 |
|----------|----------|----|------|
| LogReg | `penalty` | `elasticnet` | L1 sparse selection + L2 안정화 |
| LogReg | `solver` | `saga` | ElasticNet 지원 유일 sklearn solver |
| LogReg | `C` | `2.0` | 5-fold CV 검증 (§5 표 참고) |
| LogReg | `l1_ratio` | `0.15` | 텍스트 고차원 정설 (Zou & Hastie 2005) |
| LogReg | `max_iter` | `2000` | saga 수렴 여유 (150K 대규모 데이터) |
| LogReg | `tol` | `1e-4` | 기본값 유지 |
| ComplementNB | `alpha` | `0.5` | Laplace smoothing; sparse 안정화 |

---

## 4. Augmented Stacking

### 4.1 12 Hand-crafted Meta Features

TF-IDF가 포착하지 못하는 **문서 구조 정보**를 메타 학습기에 직접 주입한다.
4개 base 확률 + 12개 메타 피처 = **16차원 입력** (`final/202431626_김지우.ipynb §7.1 (Cell 17)`).

| 카테고리 | 피처명 | 설명 |
|----------|--------|------|
| 길이 (5) | `len_chars` | 원문 문자 수 |
| 길이 (5) | `len_morphs` | morph_view 토큰 수 |
| 길이 (5) | `len_senti` | sentiment_view 토큰 수 |
| 길이 (5) | `compression` | `len_senti / len_morphs` (의미 집중도) |
| 길이 (5) | `log_len` | `log1p(len_chars)` (길이 스케일 완화) |
| 부호 (4) | `n_excl` | 느낌표(!) 개수 |
| 부호 (4) | `n_quest` | 물음표(?) 개수 |
| 부호 (4) | `n_tilde` | 물결(~) 개수 |
| 부호 (4) | `n_dot_runs` | 줄임표(`..`) 개수 |
| 한국어 (3) | `n_kkk` | ㅋ 연속 패턴 수 |
| 한국어 (3) | `n_hhh` | ㅎ 연속 패턴 수 |
| 한국어 (3) | `n_ttt` | ㅠ/ㅜ 연속 패턴 수 |
| 부정/강조 (2) | `n_negation` | 부정어 빈도 (안/못/않/없/아니) |
| 부정/강조 (2) | `n_intensifier` | 강조어 빈도 (정말/진짜/매우/너무/아주/완전) |

### 4.2 Meta-learner — LogReg L2 (C=1.0), 16차원

후보 메타 학습기 비교 결과 단순한 LogReg L2가 최적이었다 (`final/202431626_김지우.ipynb §7.2 (Cell 17)`):

| Meta Learner | Val MCC | LB MCC | CV-LB Gap | 채택 |
|--------------|---------|--------|-----------|------|
| **LogReg L2 (C=1.0)** | **0.7456** | **0.7570** | **+0.011** | **✓ 채택** |
| HistGBT (sample_weight=1.25) | 0.7748 | 0.7500 | -0.025 | ✗ overfit |
| HistGBT (sample_weight=1.15) | 0.7745 | 0.7530 | -0.022 | ✗ overfit |

**선택 근거**: HistGBT는 val에서 +0.029 우월하나 LB에서 -0.007 열위.
30K 샘플 × 16 피처 환경에서 GBT(max_depth=6, 64개 leaf)는 sample-specific noise를 학습한다.
LogReg는 16개 weight만 학습하여 일반화 용량이 제한적 → generalization 우위.

### 4.3 StandardScaler 위치

StandardScaler는 **메타 입력(16차원)에만** 적용한다.

- base 모델 입력(TF-IDF 행렬): 비음수 sparse → 별도 스케일링 불필요
- 메타 입력: `len_chars`(수십~수백)와 `n_kkk`(0~수 개) 단위 불일치 → 스케일 통일 필수

```python
scaler = StandardScaler()
meta_X_val_scaled = scaler.fit_transform(meta_X_val)  # 메타 입력에만 적용
# ↑ scaler.fit은 val(20%)에서만 수행 (train 누수 방지)
meta_clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                               random_state=SEED, max_iter=1000)
meta_clf.fit(meta_X_val_scaled, y_val)
```

### 4.4 Threshold 0.5 — Calibrated 모델의 최적 경계

threshold sweep 실험에서 0.5가 이미 최적임을 확인하였다 (`log/threshold_search.csv`):

| Threshold | LB MCC | 비고 |
|-----------|--------|------|
| 0.490 | 0.7450 | 하락 |
| 0.495 | 0.7457 | val 최적 |
| **0.500** | **0.7456** (val) → **LB 0.757** | **채택 (LB 최적)** |
| 0.505 | LB 0.750 | 하락 |
| 0.510 | LB 0.749 | 추가 하락 |

FP=3,381 / FN=2,882 비율이 약 1.17:1 → 거의 평형 상태. threshold 변경은 zero-sum trade-off.

---

## 5. Hyperparameter Justification Table

`quiz_prep.md Q11~Q22` 근거 정리:

| 파라미터 | 채택값 | 근거 | 대안 대비 이유 |
|----------|--------|------|----------------|
| `ngram_range` (word) | `(1, 2)` | bigram이 부정 표현 "안 좋다" 포착 | `(1,1)`: CV 0.6515, 부정 미포착; `(1,3)`: +0.001 but 차원 2.5× |
| `ngram_range` (char) | `(2, 5)` | 2자~5자 한국어 감성 표현 커버 최적 | `(1,*)`: 단일 자모 의미 모호; `(6,*)`: unique 과다 → 차원 폭증 |
| `min_df` | `3` (word) / `5` (char) | 오타·1회성 고유명사 long-tail 제거 | 너무 크면 의미 있는 희귀어도 탈락 |
| `max_df` | `0.95` | 95%+ 등장 기능어 자동 필터 | 1.0이면 "이", "는" 같은 고빈도 기능어 잔류 |
| `sublinear_tf` | `True` | `1 + log(TF)` — 긴 리뷰 빈도 폭주 완화 | `False`이면 긴 리뷰에 편향 |
| `C` (LogReg) | `2.0` | 5-fold CV 검증 최적값 (§5 표 참고) | C=4.0: train-val gap 0.080 (과적합 시작) |
| `l1_ratio` | `0.15` | L1 15% sparse + L2 85% 안정 — 텍스트 정설 | 0.50: sparsity 64%, 정보 손실 우려 |
| `solver` | `saga` | ElasticNet 지원 유일 sklearn solver | `lbfgs`/`liblinear`: L2만 지원 |

**C=2.0 5-fold CV 근거 표** (단일 morph + LogReg ElasticNet, `quiz_prep.md Q19`):

| C | Val MCC | Train-Val Gap | 판단 |
|---|---------|---------------|------|
| 0.5 | 0.6779 | +0.020 | underfit |
| 1.0 | 0.6869 | +0.040 | — |
| **2.0** | **0.6905** | **+0.060** | **sweet spot** |
| 4.0 | 0.6891 | +0.080 | 과적합 시작 |
| 8.0 | 0.6815 | +0.110 | 강한 과적합 |

---

## 6. Validation Strategy

### 6.1 80/20 Single Split 선택 이유

| 방법 | 소요 시간 | 평가 안정성 | 채택 |
|------|-----------|-------------|------|
| 80/20 single split | 90분 | 30K 샘플 — 충분 | **✓** |
| 5-fold CV | 7.5시간 | 매우 안정적 | ✗ (Colab 12h 제약) |
| `cross_val_predict` | 7.5시간 | 5-fold 동일 | ✗ |

Colab 세션 12시간 한도 내에서 여러 실험을 병행하려면 5-fold는 비현실적이다.

선행 실험(M1 Baseline)에서 5-fold CV 0.7402 → LB 0.733, CV-LB gap +0.007로 작음을 확인.
이 신뢰성에 근거하여 후속 단계를 80/20 single split으로 전환하였다 (`quiz_prep.md Q10`).

### 6.2 Phase 1 (80%) → 메타 학습용 Unbiased 예측 확보

```
Phase 1:
  80% (train) ──→ base 모델 학습
                         │
  20% (val)   ──→ base 모델 예측  ←── 모델이 본 적 없는 데이터
                         │
                  unbiased val predictions
                         │
                  + meta features (12개)
                         │
                  meta_X_val (30K × 16)
                         │
                  StandardScaler → meta_clf.fit()
```

### 6.3 Phase 2 (100%) → Test 예측용 Base 재학습

```
Phase 2:
  100% (full train) ──→ base 모델 재학습 (더 많은 데이터 → test 품질 ↑)
                                │
  test ──→ base 예측 + meta features ──→ meta_clf.predict() (Phase 1에서 고정)
```

### 6.4 데이터 누수 차단 원리

만약 100%로 학습한 base 모델로 학습 데이터를 예측하면,
모델이 정답을 외운 상태라 비현실적으로 높은 확률이 생성된다.
메타 학습기는 이 가짜 신호를 학습하여 실제 test에서 무너진다.

Phase 1의 20% holdout 예측이 **OOF(Out-of-Fold) prediction** 역할을 하여 누수를 차단한다 (`quiz_prep.md Q2`).

```python
# Phase 1: 80% 학습 → 20% val 예측 (unbiased)
base_ens_80.fit(train.iloc[idx_tr][VIEW_COLS], y_full[idx_tr])
# ↑ idx_tr은 전체의 80%. val 20%는 학습에 사용 안 함.

# Phase 2: 100% 학습 → test 예측
base_ens_full.fit(train[VIEW_COLS], y_full)
# ↑ meta_clf는 Phase 1에서 학습한 것 그대로 사용 (재학습 금지)
```

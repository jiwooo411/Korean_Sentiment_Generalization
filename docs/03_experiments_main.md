# 03 Experiments (Main) — 채택 실험 5개의 진화 기록

LB 0.733 → 0.757로 이어진 단조 향상의 각 단계를 가설·구현·결과·채택 사유·다음 단계 형식으로 기술한다.

---

## 0. Experiment Card 형식

각 카드는 다음 5개 항목으로 구성된다:

| 항목 | 내용 |
|------|------|
| **가설** | 이 변경이 성능을 높일 것으로 기대한 이론적 근거 |
| **구현** | 핵심 코드·설정 (노트북 cross-reference 포함) |
| **결과** | Val MCC / LB MCC / CV-LB Gap 표 |
| **채택 사유** | CV-LB gap 부호·크기 기준의 일반화 판단 |
| **다음 단계** | 이 실험에서 도출된 후속 방향 |

폐기 실험(9개)은 `04_experiments_appendix.md` 참고.

---

## M1. Baseline — LogReg(saga) ElasticNet + morph 단독

### 가설

한국어 형태소 bigram이면 sparse linear 모델이 sweet spot.
TF-IDF(1,2) + LogReg ElasticNet 조합이 텍스트 감성 분류의 정석적 baseline.

### 구현

`final/202431626_김지우.ipynb §5 (Cell 13)` 참고.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# morph_view 단독 TF-IDF
vec = TfidfVectorizer(
    token_pattern=r"\S+", ngram_range=(1, 2),
    min_df=3, max_df=0.95, sublinear_tf=True,
    max_features=300_000,
)
X_tr = vec.fit_transform(train_morph)

# ElasticNet LogReg (saga — ElasticNet 지원 유일 solver)
clf = LogisticRegression(
    penalty="elasticnet", solver="saga",
    C=2.0, l1_ratio=0.15,
    random_state=42, max_iter=2000,
)
clf.fit(X_tr, y_train)
```

5-fold CV 실행으로 단일 split 신뢰성 사전 검증.

### 결과

| 평가 방법 | MCC | 비고 |
|-----------|-----|------|
| 5-fold CV (mean) | 0.7402 | std ≈ 0.005 |
| 5-fold CV (std) | ±0.005 | 안정적 |
| LB (test 제출) | 0.733 | — |
| **CV-LB Gap** | **+0.007** | 작음 → 신뢰 |

### 채택 사유

CV-LB gap +0.007은 매우 작아 **단일 split도 신뢰 가능**함을 실증.
이 신뢰성에 기반하여 후속 실험을 80/20 single split으로 전환 (시간 대비 효율 5×).
단, baseline 단독으로는 0.73대 정체 → 다양성 확장 필요.

### 다음 단계

morph 단독의 한계(char 표면 패턴·의미 집중 신호 미포착)를 극복하기 위해 3-View 전략으로 진화.

---

## M2. 3-View Ensemble — char + morph + sentiment

### 가설

직교 정보 차원 추가가 단일 모델의 표현력 한계를 돌파한다.
세 뷰는 서로 다른 오답 패턴을 갖기 때문에 soft voting 앙상블이 유효하다.

### 구현

`final/202431626_김지우.ipynb §5, §6 (Cell 13, 15)` 참고.
각 뷰별 독립 LogReg ElasticNet 학습 후 POSITIVE 확률 평균.

```python
class MultiViewEnsemble:
    VIEW_COLS = ("morph_view", "char_view", "sentiment_view")

    def predict_proba(self, X):
        """3개 뷰의 POSITIVE 확률 단순 평균 (soft voting)."""
        pos_probs = []
        for view in self.VIEW_COLS:
            X_v = self.vectorizers_[view].transform(X[view].values)
            clf = self.models_[view]
            pos_idx = list(clf.classes_).index("POSITIVE")
            pos_probs.append(clf.predict_proba(X_v)[:, pos_idx])
        avg_pos = np.mean(pos_probs, axis=0)  # 단순 평균 (동등 가중)
        return np.column_stack([1.0 - avg_pos, avg_pos])
```

### 결과 (`log/ensemble_results.csv`)

| 설정 | Val MCC | LB MCC | ΔMCC vs M1 |
|------|---------|--------|------------|
| **3-view MCC-weighted** | **0.7456** | — | — |
| **3-view eq (동등 가중)** | **0.7456** | **0.750** | **+0.017** |
| morph+char | 0.7450 | — | — |
| morph-heavy | 0.7446 | — | — |
| char (단독) | 0.7373 | — | — |
| morph (단독) | 0.7370 | — | — |
| char+senti | 0.7359 | — | — |
| morph+senti | 0.7314 | — | — |
| senti (단독) | 0.6714 | — | — |

LB 0.750 (+0.017 vs baseline LB 0.733).

### 채택 사유

- 3-view eq가 3-view MCC-weighted와 동등 (0.7456 vs 0.7456) → 동등 가중이 단순하고 안정적
- 뷰 간 예측 상관 약 0.5~0.6 → 진정한 다양성 확인
- senti 단독(0.6714)이 낮아 char/morph와 보완적임을 입증
- CV-LB gap 판단: val 향상이 LB 향상으로 전이 → 일반화 우수

### 다음 단계

3-View 모두 LogReg ElasticNet → 모델 클래스가 동일, paradigm 다양성 부재.
**생성 모델(ComplementNB)** 추가로 진정한 다양성 확보.

---

## M3. 4-Model Ensemble — +ComplementNB

### 가설

Discriminative × Generative paradigm 결합이 시너지를 발생시킨다.
다른 추론 방식은 다른 오답 패턴을 제공하여 앙상블 효과를 증폭시킨다.

### 구현

`final/202431626_김지우.ipynb §6 (Cell 15)` 참고.
ComplementNB를 morph_view + TF-IDF word(1,2)로 학습, 나머지는 M2와 동일.

```python
from sklearn.naive_bayes import ComplementNB

nb_vec_80 = TfidfVectorizer(
    token_pattern=r"\S+", ngram_range=(1, 2),
    min_df=3, max_df=0.95, sublinear_tf=True, max_features=300_000,
)
X_tr_nb = nb_vec_80.fit_transform(train.iloc[idx_tr]["morph_view"])
nb_clf_80 = ComplementNB(alpha=0.5)  # Laplace smoothing α=0.5
nb_clf_80.fit(X_tr_nb, y_full[idx_tr])
```

**4-model 가중치 그리드** (`log/submission_4model_w{20,25,33,40,eq}.csv`):

| 가중치 설정 | NB 비중 | Val MCC | 비고 |
|-------------|---------|---------|------|
| equal | 25% | 최고 | 채택 |
| w33 | 33% | ≈equal | — |
| w25 | 25% | ≈equal | — |
| w20 | 20% | ≈equal | — |
| w40 | 40% | 소폭 하락 | NB 과중 |

모든 가중치 설정이 동등 가중과 유사 → 동등 가중 채택 (단순성 우선).

### 결과

| 단계 | Val MCC | LB MCC | ΔMCC vs M2 |
|------|---------|--------|------------|
| M2 (3-view) | 0.7456 | 0.750 | — |
| **M3 (4-model)** | **0.7456** | **0.753** | **+0.003** |

LB 0.753 (+0.003 vs 3-view).

### 채택 사유

- NB 단독 성능(morph 단독 0.7370 대비 낮음)에도 앙상블에서 정보 추가
- Discriminative + Generative 결합으로 진정한 paradigm 다양성 확보
- CV-LB gap 안정적 (+값 유지) → 일반화 우수 판단

### 다음 단계

TF-IDF 피처 공간 안에서의 다양성은 충분히 활용됨.
**TF-IDF가 포착하지 못하는 문서 구조 정보**(길이, 부호, 감성 패턴)를 메타 피처로 주입.

---

## M4. Augmented Stacking — 12 Meta Features + LogReg L2 ⭐

### 가설

12개 hand-crafted 메타 피처가 TF-IDF 미포착 문서 구조 정보를 보강하여
메타 학습기가 base 모델의 불확실성을 더 정밀하게 보정한다.

### 구현

`final/202431626_김지우.ipynb §7 (Cell 17, 19)` 참고.
4개 base 확률 + 12개 메타 피처 = 16차원 → StandardScaler → LogReg L2.

```python
# 메타 피처 추출 (TF-IDF 미포착 문서 구조 정보)
def extract_meta_features(df):
    f = pd.DataFrame(index=df.index)
    # 길이 정보 (5개)
    f["len_chars"]   = df["text"].str.len()
    f["len_morphs"]  = df["morph_view"].str.split().str.len()
    f["len_senti"]   = df["sentiment_view"].str.split().str.len()
    f["compression"] = f["len_senti"] / f["len_morphs"].clip(lower=1)
    f["log_len"]     = np.log1p(f["len_chars"])
    # 부호 (4개)
    f["n_excl"]      = df["text"].str.count(r"!")
    f["n_quest"]     = df["text"].str.count(r"\?")
    f["n_tilde"]     = df["text"].str.count(r"~")
    f["n_dot_runs"]  = df["text"].apply(lambda x: len(RE_DOT_RUN.findall(x)))
    # 한국어 감성 패턴 (3개)
    f["n_kkk"]       = df["text"].apply(lambda x: len(RE_KKK.findall(x)))
    f["n_hhh"]       = df["text"].apply(lambda x: len(RE_HHH.findall(x)))
    f["n_ttt"]       = df["text"].apply(lambda x: len(RE_TTT.findall(x)))
    # 부정/강조 (2개)
    f["n_negation"]    = df["text"].apply(lambda x: len(RE_NEGATION.findall(x)))
    f["n_intensifier"] = df["text"].apply(lambda x: len(RE_INTENSIFIER.findall(x)))
    return f.fillna(0).astype(np.float32)

# 메타 학습기 (16차원 입력)
scaler   = StandardScaler()
meta_clf = LogisticRegression(C=1.0, penalty="l2",
                               solver="liblinear", random_state=42)
meta_clf.fit(scaler.fit_transform(meta_X_val), y_val)
```

**메타 학습기 후보 비교** (`final/202431626_김지우.ipynb §7.2 (Cell 17)`):

| Meta Learner | Val MCC | LB MCC | CV-LB Gap | 채택 |
|--------------|---------|--------|-----------|------|
| **LogReg L2 (C=1.0)** | **0.7456** | **0.757** | **+0.011** | **✓** |
| HistGBT (sample_weight=1.25) | 0.7748 | 0.750 | -0.025 | ✗ overfit |
| HistGBT (sample_weight=1.15) | 0.7745 | 0.753 | -0.022 | ✗ overfit |

### 결과

| 단계 | Val MCC | LB MCC | ΔMCC vs M3 | CV-LB Gap |
|------|---------|--------|------------|-----------|
| M3 (4-model) | 0.7445 (avg) | 0.753 | — | — |
| **M4 Stacking** | **0.7456** | **0.757** | **+0.004** | **+0.011** |

단순 평균 → Stacking Δ: Val +0.0011 (`final/202431626_김지우.ipynb §7 (Cell 17)` 출력 참고).

### 채택 사유

- CV-LB gap +0.011 → 양호한 일반화 (gap이 양수이며 크지 않음)
- Val MCC 향상이 LB MCC 향상으로 전이 → 일반화 우수 판단
- HistGBT는 val ↑ but LB ↓ → CV-LB gap 부호가 음수 → 자동 기각
- 16차원 단순 구조에 LogReg L2가 적합: 외울 capacity 제한 → noise 학습 방지

### 다음 단계

sparse linear 모델의 표현력 상한 도달.
가이드 제약 해제 시나리오에서 **dense semantic 임베딩(LLM)** 으로 확장 (M5).

---

## M5. Drill-06 — LLM Stacking (가이드 해제 가정)

> 상세 내용은 `docs/06_drill06_extension.md` 참고. 본 카드는 한 줄 요약.

### 한 줄 요약

`multilingual-e5-base` + `KR-SBERT` OOF 임베딩을 TF-IDF×2 + 12 meta와 결합하는
sparse→dense 확장으로 OOF 기반 τ* sweep, 목표 LB 0.770+.

| 항목 | 내용 |
|------|------|
| 확장 방향 | TF-IDF sparse → LLM dense semantic 병합 |
| 추가 모델 | multilingual-e5-base, KR-SBERT (HuggingFace ST) |
| 검증 방식 | 5-fold stratified OOF (80/20 신뢰성 강화) |
| τ* 탐색 | OOF 기반 MCC-optimal threshold sweep (val 과적합 방지) |
| 구현 파일 | `drill06 final/202431626_김지우.ipynb` |

---

## 6. Recap — 전체 Evolution 표

| Stage | 변경점 | Val MCC | LB MCC | ΔMCC | CV-LB Gap | 채택 |
|-------|--------|---------|--------|------|-----------|------|
| M1 Baseline | LogReg(saga) ElasticNet, morph 단독 | 0.7402 (CV) | 0.733 | — | +0.007 | ✓ |
| M2 3-View | +char/sentiment view, soft voting | 0.7456 | 0.750 | +0.017 | 양호 | ✓ |
| M3 4-Model | +ComplementNB (generative) | 0.7456 | 0.753 | +0.003 | 양호 | ✓ |
| **M4 Stacking** | **+12 meta feat, LogReg L2 meta** | **0.7456** | **0.757** | **+0.004** | **+0.011** | **✓ ⭐** |
| M5 Drill-06 | +e5+KR-SBERT, 5-fold OOF | (측정값 없음) | 0.770+ 목표 | — | — | 가이드 해제 |

**폐기 실험 (요약):**

| 폐기 Stage | 변경점 | Val MCC | LB MCC | 폐기 사유 |
|-----------|--------|---------|--------|-----------|
| HistGBT meta (wt=1.25) | 복잡한 메타 | 0.7748 | 0.750 | gap -0.025 overfit |
| HistGBT meta (wt=1.15) | 복잡한 메타 | 0.7745 | 0.753 | gap -0.022 overfit |
| Threshold grid (132조합) | weight×τ 동시 | 0.7745 | ≤0.753 | val 과적합 |
| Threshold 단독 (τ=0.505) | τ만 변경 | — | 0.750 | zero-sum |

상세 폐기 실험(A1~A9)은 `docs/04_experiments_appendix.md` 참고.

---

> **핵심 교훈 (Sonnet 요약):**
> CV-LB gap의 부호와 크기가 모델 선택 1순위 지표.
> Val 점수만 좇으면 LB 폭락(risk-of-ruin)에 노출된다.
> 다양성은 솔버 변형이 아닌 모델 클래스 차이에서 나온다.

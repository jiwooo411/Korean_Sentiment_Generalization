# Drill-06 LLM Embedding Extension

sparse → dense semantic 확장 트랙: TF-IDF 기반 lexical 앙상블에 multilingual embedding을 결합해 LB 0.757 → 0.770+ 달성을 목표로 한 파이프라인 설계.

---

## 1. 동기 (가이드 §2 해제 가정)

### Drill-05 천장

| 지표 | 값 |
|---|---|
| Drill-05 LB MCC | 0.757 |
| NSMC SOTA (sklearn-only 제약 해제 참고) | ~0.77 |
| 잔여 gap | ~0.013 |

Drill-05는 `sklearn-only` 제약 하에서 TF-IDF 기반 sparse linear 분류기로 LB 0.757, Rank 5에 도달했다. 그러나 추가적인 단조 증가를 가로막는 두 가지 천장이 존재한다.

### 한계 분석

**Lexical ceiling**

TF-IDF bigram은 `(1,2)` n-gram으로 부정 표현을 포착한다. 예를 들어 "안 좋다"는 bigram `["안 좋다"]`로 표현되어 단일 NEGATIVE 신호로 동작한다. 그러나 paraphrase와 synonym은 포착하지 못한다.

- "재미없다" vs "지루하다": 두 표현은 유사한 의미이지만 TF-IDF 공간에서 전혀 다른 좌표에 위치
- "최고예요" vs "정말 좋았어요": 동일 감성의 paraphrase — lexical 모델은 별개로 학습

**확장 가설**

cross-paradigm 다양성(lexical × semantic)을 도입하면 두 모델군의 예측 오류가 상관 0.5~0.6으로 낮게 유지되어 앙상블 시너지 극대화 가능.

- intra-lexical 상관: ~0.8+ (공통 어휘 공간)
- intra-semantic 상관: ~0.7+ (동일 transformer paradigm이지만 학습 task/데이터 다름)
- cross-group 상관: **~0.5~0.6** → 진정한 앙상블 시너지 source

---

## 2. 환경 (Colab T4 GPU)

### Drill-05 대비 변경점

| 항목 | Drill-05 | Drill-06 |
|---|---|---|
| 하드웨어 | CPU (런타임 제한 없음) | **Colab T4 GPU** (drill-06-notice 권장) |
| 추가 라이브러리 | sklearn, pandas, numpy, pecab, tqdm | + `sentence-transformers`, `torch` |
| 인코딩 시간 (15만 문장) | N/A | T4 기준 분 단위 완료 |
| 재현성 seed | 42 | 42 (동일 원칙) |

T4 GPU 없이 CPU로 실행하면 multilingual-e5-base 인코딩에 수 시간 소요. GPU 환경 필수.

---

## 3. 모델 구성 (4-base + meta)

### Base 모델 4개

| 모델 ID | dim | 학습 paradigm | 채택 이유 |
|---|---|---|---|
| `tfidf_word` | ~100K sparse | Discriminative / Lexical | Drill-05 핵심 모델. word bigram으로 부정 표현 포착. |
| `tfidf_char` | ~100K sparse | Discriminative / Lexical | `char_wb (2,5)` — pecab 미포착 typo/이모티콘 표면 패턴 자동 학습. |
| `intfloat/multilingual-e5-base` | 768d dense | Semantic / Retrieval-tuned | retrieval task 학습, paraphrase robustness. 다국어 corpus → 일반화 강함. |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 768d dense | Semantic / NLI-STS-tuned | 한국어 NLI/STS fine-tuned. e5와 학습 task/domain 다름 → diversity 확보. |

**Meta learner**: LogReg L2 (C=1.0) — 14d 저차원 입력에 ElasticNet sparsity 불필요.

### 조합 전략

두 embedding 모델이 동일 family(e5-base + e5-large)라면 상관이 너무 높아 ensemble 효과 미미. multilingual-e5(retrieval) vs KR-SBERT(NLI)는 학습 paradigm과 도메인이 달라 **decision boundary diversity** 확보.

---

## 4. Embedding 설계

### E5 prefix 규칙

```python
# intfloat/multilingual-e5-base: 분류 task에는 "query: " prefix 필수
texts_e5 = ["query: " + t for t in texts]

# snunlp/KR-SBERT-V40K-klueNLI-augSTS: prefix 불필요
texts_krsbert = texts  # raw text 그대로
```

E5 계열은 `"query: "` / `"passage: "` prefix를 학습 시 사용했기 때문에, 추론 시 prefix 없이 입력하면 성능 저하. 분류 task에는 `"query: "` 사용.

### max_len 설정

```python
MAX_LEN = 128  # 영화 리뷰 p99 ~120자 → 128 충분
```

EDA 결과 (노트북 Cell §3.2):
- 평균 길이: 35자
- p99 길이: ~120자

max_len=128로 99%+ 리뷰를 손실 없이 커버. 256으로 늘려도 성능 변화 미미하나 메모리/시간 2배.

### npy 캐시 전략

```python
CACHE_DIR = "log/drill06_cache/"
# 경로 예시
# log/drill06_cache/emb_e5_train.npy
# log/drill06_cache/emb_e5_test.npy
# log/drill06_cache/emb_krsbert_train.npy
# log/drill06_cache/emb_krsbert_test.npy
```

인코딩 1회 비용이 크므로 numpy `.npy`로 disk cache. 재실행 시 파일 존재 여부 확인 후 skip.

---

## 5. Validation 강화 (단일 split → 5-fold stratified OOF)

### Drill-05 단일 split의 한계

Drill-05의 80/20 single split은 30K val 샘플으로 메타 학습기 검증에 충분했다. 그러나 Drill-06에서는 다음 복잡성이 추가된다.

- 4-base 모델 (2 lexical + 2 semantic)
- 10 hand-crafted meta feature
- MCC-optimal threshold sweep

이 세 조작이 동일 val set에서 동시에 평가되면 **다중 비교 과적합(multiple comparison overfitting)** 위험 증가.

### OOF의 구조적 보장

| 방식 | 누수 위험 | 시간 | 샘플 효율 |
|---|---|---|---|
| 80/20 single split | 중간 (val이 사실상 두 번째 train) | 1× | val 30K |
| **5-fold stratified OOF** | **없음** (각 샘플이 fold 외부 모델로 예측) | 5× | train 전체 사용 |

OOF 원칙: 각 샘플의 예측은 **그 샘플을 학습에 사용하지 않은 fold 모델**이 생성 → 진정한 out-of-sample 예측.

### Leakage 차단 규칙

```python
for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_text, y_train)):
    # vectorizer/scaler는 반드시 fold 내부 train split에서만 fit
    vec_word = TfidfVectorizer(...).fit(X_train_text[tr_idx])  # ← fold마다 새로 fit
    scaler_e5 = StandardScaler().fit(emb_e5_train[tr_idx])    # ← fold마다 새로 fit
    # val에는 transform만
    X_val_word = vec_word.transform(X_train_text[val_idx])
```

vectorizer/scaler를 전체 train에서 fit하면 val 통계가 누수됨. fold마다 새로 fit 필수.

---

## 6. Hand-crafted Meta Features (10개)

### Drill-05 12개 → Drill-06 10개 압축

Drill-05에서 12개 meta feature를 사용했으나, Drill-06에서는 embedding 2개가 추가되어 전체 meta 차원이 늘어난다. 중복성이 높은 feature를 정리해 10개로 압축.

### Feature 목록

| Feature | 카테고리 | 의미 |
|---|---|---|
| `len_chars` | 길이 | 문자 수 |
| `log_len` | 길이 | log(len+1) — 길이 outlier 완화 |
| `n_excl` | 부호 | `!` 개수 |
| `n_quest` | 부호 | `?` 개수 |
| `n_dot_runs` | 부호 | `...` 개수 |
| `n_kkk` | 한국어 패턴 | `ㅋ` 반복 (POSITIVE marker) |
| `n_ttt` | 한국어 패턴 | `ㅠ/ㅜ` 반복 (NEGATIVE/강조) |
| `n_hhh` | 한국어 패턴 | `ㅎ` 반복 (POSITIVE marker) |
| `n_negation` | 언어 | 안/못/않/없/아니 출현 횟수 |
| `n_intensifier` | 언어 | 정말/진짜/매우/너무/완전 |

### Meta 입력 차원

```
4 base OOF proba + 10 meta features = 14d
```

Drill-05 16d (4 base + 12 meta) → Drill-06 14d (4 base + 10 meta). base 구성이 동일하므로 meta feature 수 감소로 차원 절감.

---

## 7. MCC-optimal Threshold Sweep

### 전략 개요

```python
thresholds = np.linspace(0.30, 0.70, 81)  # 0.005 step, 81 candidates
```

### OOF에서만 sweep — Drill-05 실패 교훈 반영

Drill-05에서 threshold grid search의 실패 사례:

| 시도 | Val MCC | LB MCC | 교훈 |
|---|---|---|---|
| HistGBT meta + threshold grid (132조합) | 0.7748 | 0.750 | val 과적합 |
| threshold 단독 0.505 | — | 0.750 | zero-sum |
| threshold 단독 0.510 | — | 0.749 | zero-sum |

**Drill-06 설계**: threshold sweep을 **OOF prediction에서만** 수행. test 정보 절대 미사용.

```python
# 올바른 방법: OOF proba에서 sweep
best_tau, best_mcc = 0.5, -1
for tau in np.linspace(0.30, 0.70, 81):
    preds = (oof_proba[:, 1] >= tau).astype(int)
    mcc = matthews_corrcoef(y_train_enc, preds)
    if mcc > best_mcc:
        best_mcc, best_tau = mcc, tau
```

### Safety Net

τ* 와 τ=0.5의 MCC 차이가 **0.002 미만**이면 τ=0.5 사용. 작은 개선은 노이즈일 가능성 높음.

### 시각화

최적 threshold 곡선: `log/drill06_cache/threshold_curve.png`

---

## 8. Base Diversity 분석

### OOF Proba Correlation 해석

| 모델 쌍 | 예상 상관 | 해석 |
|---|---|---|
| tfidf_word × tfidf_char | ~0.8+ | intra-lexical. char_wb가 typo 포착이므로 보완성 존재 |
| emb_e5 × emb_krsbert | ~0.7+ | intra-semantic. 학습 corpus 달라 완전 중복 아님 |
| **lexical × semantic (cross-group)** | **~0.5~0.6** | **핵심 diversity source. 앙상블 시너지 대부분 여기서 발생** |

### Ensemble 시너지 원리

두 모델 군의 오류 패턴이 다를 때 앙상블이 작동한다.

- Lexical 모델 오류: synonym 미포착 ("재미없다" vs "지루하다"가 다른 신호)
- Semantic 모델 오류: rare entity, 신조어 (pre-training에 없는 어휘)

cross-group 상관 0.5~0.6은 앙상블 효과가 유의미하게 나타나는 범위. 0.9+ 이면 ensemble 효과 미미.

---

## 9. Drill-05와의 비교 표

| 항목 | Drill-05 | Drill-06 | 변화 이유 |
|---|---|---|---|
| 검증 방식 | 80/20 single split | **5-fold stratified OOF** | 다중 조작 동시 평가 시 val 과적합 위험 증가 |
| Base 모델 구성 | 4× sklearn (LogReg×3 + NB) | **2× sklearn + 2× embedding** | lexical × semantic 직교성으로 cross-paradigm diversity 확보 |
| Meta 입력 차원 | 4 base + 12 meta = 16d | **4 base + 10 meta = 14d** | embedding 추가로 base 표현력 상승 → meta feature 압축 |
| Threshold 방식 | 0.5 고정 (grid search 모두 실패) | **OOF τ* sweep (0.30~0.70, 81개)** | OOF 한정으로 val 과적합 방지하면서 MCC 최적화 |
| 환경 | CPU only | **Colab T4 GPU** | 768d embedding 15만 문장 인코딩 — GPU 필수 |
| 허용 라이브러리 | sklearn-only | **+ sentence-transformers, torch** | 가이드 §2 해제 가정 (drill-06-notice) |
| 목표 LB MCC | 0.757 (달성) | **0.770+** | sparse → dense semantic 확장으로 gap 감소 |

---

## 10. CV-LB Gap 전망 + Submission Strategy

### Gap 전망

Drill-05 실측 CV-LB gap:

| | Val MCC | LB MCC | Gap | 해석 |
|---|---|---|---|---|
| LogReg L2 meta (최종) | 0.7456 | **0.7570** | **+0.011** | train이 더 어려움. 양의 gap = 일반화 양호 |
| HistGBT meta | 0.7748 | 0.7500 | -0.025 | 음의 gap = 명백한 overfit |

Drill-06 예상:

- OOF MCC 0.770 달성 시 → LB 예상 **0.775~0.785**
- OOF + 0.005~0.015 범위 (동일 데이터 분포 가정)

### Submission Strategy

1. **Primary**: `final_pipeline.pkl` 기반 submission (본 pipeline 전체)
2. **Safety backup**: τ=0.5 버전 (threshold 없이 0.5 고정)
3. **절대 금지**: LB score를 보고 weight/threshold 재조정 → public LB overfit 직행

```python
# 금지 패턴 예시
# LB 결과를 보고 다음을 조정하면 안 됨:
# - meta learner의 C 값
# - base model의 ensemble weight
# - threshold τ
# → public LB 50% 데이터에 과적합, private LB에서 rank 폭락
```

---

## 11. 한계 & 후속

### 의도적 한계 (설계 결정)

| 한계 | 이유 |
|---|---|
| Embedding fine-tuning 미수행 | overfit 위험 + 본 과제 가이드 범위 초과 |
| LightGBM/XGBoost meta 미사용 | Drill-05에서 GBT meta learner가 LB drop (-0.007) 확인 |
| 단일 seed 실험 | Colab 세션 시간 제약 |

### 가능한 후속 확장

| 확장 방향 | 예상 효과 | 조건 |
|---|---|---|
| `BAAI/bge-m3` 추가 | lexical×semantic diversity 추가 | 메모리 16GB+ |
| `intfloat/e5-large` 교체 | 768d → 1024d 표현력 | GPU 메모리 여유 필요 |
| Instruction-tuned model (e5-mistral) | 더 강한 의미 표현 | 연산 비용 큼 |
| Cross-lingual augmentation | 한영 병렬 데이터로 robustness ↑ | 추가 데이터 필요 |

---

> 모든 수치 및 설계 결정은 `drill06 final/202431626_김지우.ipynb` markdown 셀 §1~§16에서 인용.
> 실험 로그: `log/drill06_cache/`, threshold 시각화: `log/drill06_cache/threshold_curve.png`.

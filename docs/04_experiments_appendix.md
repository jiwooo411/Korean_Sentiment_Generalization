# 폐기 실험 전체 기록 (Appendix)

최종 모델 LB 0.757 도달 과정에서 시도했으나 메인 파이프라인에서 제거된 실험 9개의 상세 기록.
각 실험의 시도 내용, 원본 데이터 인용, 폐기 이유, 코드 연결을 정리하여 재현·학습·퀴즈 방어 자료로 활용.

---

## 폐기 실험 한눈에 보기

| Stage | # | 실험명 | Val / CV MCC | LB MCC | 채택 | 폐기 사유 |
|---|---|---|---|---|---|---|
| 1 | A1 | NEG_용언 결합 + POS 화이트리스트 | 0.6915 (CV) | — | ✗ | 과전처리 → 어미 정보 손실, baseline 정체 |
| 2 | A2 | SGD Grid v1 (alpha=1e-4~1e-3) | 0.6450 (CV) | — | ✗ | alpha 25배 과정규화, 전 조합 underfit |
| 3 | A3 | SGD Grid v2 (average=True) | 0.7278 (CV) | — | ✗ | Polyak averaging이 L1 sparsity 무력화 |
| 4 | A4 | SGD Grid v3 (average=False) | 0.7391 (CV) | — | ✗ | LogReg와 통계 동등(Δ 0.001), 앙상블 가치 없음 |
| — | A5 | Snapshot Ensemble (확률 평균) | — | — | — | LB 0.757 확정 후 미시도 (한계 효용) |
| 8 | A6 | HistGBT Meta (sample_weight=1.25) | 0.7748 (Val) | 0.750 | ✗ | Val 과적합, CV-LB gap −0.025 |
| 8 | A7 | HistGBT Meta (sample_weight=1.15) | 0.7745 (Val) | 0.753 | ✗ | 동일 원인, weight 완화로 미해결 |
| 9 | A8 | Threshold Recalibration Grid (132조합) | 0.7745 (Val) | 0.752–0.753 | ✗ | Val grid 과적합, 전 조합 LogReg meta 미달 |
| 10 | A9 | Threshold Tuning 단독 (LogReg meta) | — | 0.749–0.750 | ✗ | 0.5에서 이미 calibrated, zero-sum trade-off |

---

## 실험 A1: NEG_용언 결합 + POS 화이트리스트 전처리

### 무엇을 했나

- 형태소 분석 후 부정어("안", "못", "않")가 용언 앞에 오면 단일 토큰으로 결합
  - 예: `["안", "좋다"]` → `["NEG_좋다"]`
- 9개 품사 화이트리스트 (NNG/NNP/VV/VA/MAG/MAJ/IC/SL/XR)로 강한 POS 필터링
- 반복 자모 (`ㅋㅋㅋㅋ`) → `KKK`, (`ㅠㅠㅠ`) → `TTT` 심볼 치환

### 결과

| 설정 | CV MCC |
|---|---|
| baseline (형태소 단독) | 0.6886 |
| NEG_용언 결합 (peak) | 0.6915 |

단일 모델로 0.69에서 정체. LB 제출 없음.

### 왜 폐기했는가

**복잡한 전처리가 단순 baseline을 능가하지 못함.**

원인 분석:

1. **POS 필터링이 어미(EC/EF/ETM) 제거** → 시제·존칭 정보 소실
   - "지루했어요"(완곡 서술) vs "지루해"(반말) 분포 차이가 어미에 담김
2. **NEG_ 결합으로 토큰 희소성 증가** → `min_df=3` 필터에서 대량 탈락
3. **`KKK`/`TTT` 심볼이 희소 심볼만 양산** → 정보 가치 없음

### 교훈

> "전처리는 최소로. TF-IDF의 `min_df`/`max_df`에 필터링을 위임하라."

대체 방안: 블랙리스트 정제 + 형태소 전체 보존 → **3-View 전략**으로 진화.

### Code reference

- 실험 코드: `final/202431626_김지우.ipynb` Cell 3–4 (전처리 함수 비교)
- 3-View 대체 구현: 동 노트북 Cell 8–10

---

## 실험 A2: SGD Grid Search v1 (alpha=1e-4~1e-3)

### 무엇을 했나

LogReg(saga)의 폴드당 학습 시간 ~20분을 단축하기 위해 SGDClassifier로 대체 시도.

```python
ALPHA_GRID    = [1e-4, 5e-4, 1e-3]   # 3개
L1_RATIO_GRID = [0.15, 0.30, 0.50]   # 3개
# 9 combos × 5-fold = 45 fits
```

### 결과 (log/sgd_grid_results.csv 인용)

전체 9개 조합 중 MCC 상위 행:

| alpha | l1_ratio | train_mcc | val_mcc | sparsity | iter |
|---|---|---|---|---|---|
| 0.0001 | 0.15 | 0.6520 | **0.6450** | 0.962 | 13 |
| 0.0001 | 0.30 | 0.6356 | 0.6303 | 0.983 | 13 |
| 0.0001 | 0.50 | 0.6227 | 0.6193 | 0.991 | 13 |
| 0.0005 | 0.15 | 0.5591 | 0.5567 | 0.991 | 9 |
| 0.0010 | 0.15 | 0.5107 | 0.5093 | 0.996 | 8 |

- Best: alpha=0.0001, l1=0.15 → **val_mcc=0.6450** (LogReg 0.7402 대비 −0.095)
- 전 조합 sparsity 96–99% (피처 거의 소멸), iter 8–13에서 조기 수렴

### 왜 폐기했는가

**alpha 스케일 부정합.** LogReg(C=2.0)과의 등가 환산:

```
SGD alpha = 1 / (C × n_samples)
          = 1 / (2.0 × 120,000)
          ≈ 4e-6
```

탐색한 최소값 alpha=1e-4도 등가 alpha 대비 **25배 과잉 정규화** → 모델이 데이터를 학습하지 못하고 스파스 벡터만 반환.

### 교훈

> "솔버 교체 시 등가 환산 공식 확인 필수. SGD alpha = 1/(C·N)."

대체 방안: SGD Grid v2 (탐색 범위 1e-6~1e-4로 재설정).

### Code reference

- 원본 로그: `log/sgd_grid_results.csv` (9행 × 11열)
- 학습 코드: `final/202431626_김지우.ipynb` Cell 6 (SGD grid 실험 블록)

---

## 실험 A3: SGD Grid v2 (average=True)

### 무엇을 했나

v1의 alpha 범위를 1e-6~1e-4로 수정하고 Polyak averaging (`average=True`) 추가.

```python
ALPHA_GRID    = [1e-6, 5e-6, 2e-5, 1e-4]
L1_RATIO_GRID = [0.15, 0.30, 0.50]
# 12 combos, average=True
```

### 결과 (log/sgd_grid_v2_results.csv 인용 — sparsity=0 확인)

| alpha | l1_ratio | val_mcc | sparsity | gap |
|---|---|---|---|---|
| 1e-06 | 0.15 | 0.7278 | **0.0** | +0.155 |
| 1e-06 | 0.30 | 0.7102 | **0.0** | +0.119 |
| 1e-06 | 0.50 | 0.6990 | **0.0** | +0.106 |
| 5e-06 | 0.15 | 0.6703 | **0.0** | −0.030 |
| 5e-06 | 0.30 | 0.5902 | **0.0** | −0.093 |

전 12개 조합에서 **sparsity = 0.0%** — L1 선택 기능 완전 소멸.

- Best: alpha=1e-6, l1=0.15 → val_mcc=0.7278 (LogReg 0.7402 대비 −0.012)
- val > train (gap 음수) 조합 다수 발생 → 비정상 학습 신호

### 왜 폐기했는가

**`average=True`가 ElasticNet의 sparse selection을 무력화.**

작동 원리:
- L1이 어떤 weight를 0으로 만들어도, Polyak averaging이 전 iteration의 weight를 평균
- 0이었던 weight가 이전 iter 평균값으로 부활 → weight 0이 사라짐
- 결과: ElasticNet 이름만 있고 실제로는 약한 정규화 SGD로 동작

### 교훈

> "옵션 추가 전 다른 옵션과의 상호작용 확인 필수."
> "`average=True` × ElasticNet → L1 무력화."

대체 방안: SGD Grid v3 (`average=False`).

### Code reference

- 원본 로그: `log/sgd_grid_v2_results.csv` (12행, sparsity 열 전부 0.0)
- 상호작용 분석: `appendix.md §실험 3`

---

## 실험 A4: SGD Grid v3 (average=False)

### 무엇을 했나

v2에서 `average=False`로 수정. 나머지 설정 동일.

```python
ALPHA_GRID    = [1e-6, 5e-6, 2e-5, 1e-4]
L1_RATIO_GRID = [0.15, 0.30, 0.50]
# 12 combos, average=False
```

### 결과 (log/sgd_grid_v3_results.csv 인용)

| alpha | l1_ratio | val_mcc | sparsity | gap |
|---|---|---|---|---|
| **5e-06** | **0.15** | **0.7391** | 0.403 | +0.091 |
| 1e-06 | 0.15 | 0.7373 | 0.161 | +0.193 |
| 5e-06 | 0.30 | 0.7372 | 0.642 | +0.080 |
| 1e-06 | 0.30 | 0.7362 | 0.286 | +0.196 |
| 5e-06 | 0.50 | 0.7349 | 0.801 | +0.069 |

- Best: **alpha=5e-6, l1=0.15** → **val_mcc=0.7391**, sparsity=40.3%
- ElasticNet 정상 작동 확인 (sparsity 복원)

수학적 일관성 검증:
```
SGD alpha=5e-6 → 등가 LogReg C ≈ 1/(5e-6 × 120K) ≈ 1.67
LogReg 최적 C=2.0 과 매우 근접 — 두 옵티마이저가 같은 수렴 영역
```

LogReg(saga) val_mcc=0.7402 대비 **Δ=0.001** (검증 노이즈 범위).

### 왜 폐기했는가

**LogReg와 통계적 동등** → 앙상블 멤버 추가 가치 없음.

- 예측 상관 ≈0.95 (같은 모델 클래스: ElasticNet Logistic Regression)
- 두 모델 평균 → 시너지 없음, 연산 비용만 추가

### 교훈

> "다양성은 옵티마이저 차이가 아닌 **모델 클래스/데이터 표현 차이**에서 나온다."

대체 방안: ComplementNB 추가 (생성 모델 — 진짜 패러다임 다양성).

### Code reference

- 원본 로그: `log/sgd_grid_v3_results.csv` (12행, best: alpha=5e-6, val_mcc=0.7391)
- 앙상블 상관 분석: `final/202431626_김지우.ipynb` Cell 18 (다양성 검토 블록)

---

## 실험 A5: Snapshot Ensemble (확률 평균)

### 무엇을 했나

이미 제출된 LB 결과들의 확률을 사후 가중 평균하여 새 submission 생성:

```python
total = 0.757 + 0.753
combined_proba = (0.757 / total) * p_logreg_meta \
               + (0.753 / total) * p_4model
```

### 결과

**시도하지 않음.** LB 0.757 확정 후 안전 우선 결정.

### 왜 시도 안 했는가

- LogReg meta(0.757)가 4-model(0.753)을 포함하는 superset 구조
- 두 모델 예측 상관도 매우 높음 → 평균 효과 미미 예상
- 노트북 정리 시간 확보 우선

### 교훈

> "한계 효용 체감. 0.757에서 0.001 추가를 위해 위험 감수 안 함."

### Code reference

- 실험 구상 노트: `appendix.md §실험 5`

---

## 실험 A6: HistGBT Meta (sample_weight=1.25)

### 무엇을 했나

LogReg meta가 `n_negation` 등 hand-crafted 피처에 가중치 ~0을 부여해 무시한다는 진단:

- 비선형 메타 학습기 `HistGradientBoostingClassifier`로 교체
- NEGATIVE 샘플 25% 가중치 부여 (FP 페널티 강화)

```python
meta_gbt = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_depth=6,
    l2_regularization=1.0,
    early_stopping=True,
)
meta_gbt.fit(
    meta_X_val, y_val,
    sample_weight=np.where(y_val == 'NEGATIVE', 1.25, 1.0)
)
```

메타 학습 데이터 규모: **30,000 샘플 × 16 피처**
관련 파일: `log/submission_recal_r1_w114_t455.csv` (r1 제출)

### 결과

| 지표 | 값 |
|---|---|
| Val MCC | **0.7748** (+0.029 vs LogReg meta) |
| LB MCC | **0.750** (−0.007 vs LogReg meta) ⚠ |
| Val Recall | 0.842 (−0.031 손상) |
| Val Precision | 0.902 (+0.018) |
| CV-LB Gap | **−0.025** (적신호) |

### 왜 폐기했는가

**Bias-Variance Trade-off의 교과서적 사례.**

- HistGBT max_depth=6 → 최대 2^6=64 leaf
- 30K / 64 ≈ **468 샘플/leaf** → leaf별 sample-specific noise 학습 시작
- LogReg는 16개 weight만 학습 → 외울 capacity 부족 → 일반화 우수

CV-LB gap 비교:

| 메타 학습기 | Val MCC | LB MCC | Gap |
|---|---|---|---|
| LogReg L2 (채택) | 0.7456 | 0.7570 | **+0.011** ✓ |
| HistGBT (1.25) | 0.7748 | 0.7500 | **−0.025** ✗ |

Gap 부호 음수 → **절대적 적신호. Val 점수 단독으로 모델 선택 금지.**

### 교훈

> "Val 점수만 보고 모델 선택하지 말 것. CV-LB gap이 양호한 모델이 진짜 일반화."
> "메타 학습 데이터 크기에 맞는 모델 복잡도 선택 필수."

### Code reference

- 실험 코드: `final/202431626_김지우.ipynb` Cell 22–23 (HistGBT meta 블록)
- 제출 파일: `log/submission_recal_r1_w114_t455.csv`
- 메타 학습 입력: `log/meta_X_val_backup.csv`, `log/y_val_backup.csv`

---

## 실험 A7: HistGBT Meta (sample_weight=1.15)

### 무엇을 했나

실험 A6에서 sample_weight=1.25가 과도하다는 진단 → 1.15로 완화.

```python
sample_weight=np.where(y_val == 'NEGATIVE', 1.15, 1.0)
```

관련 파일: `log/submission_recal_r2_w114_t460.csv`, `log/submission_recal_r3_w114_t435.csv`

### 결과

| 지표 | 값 |
|---|---|
| LB MCC | **0.753** |
| 비교 | LogReg meta 0.757 대비 −0.004 |

sample_weight=1.25 (LB 0.750) 대비 미세 개선이나 LogReg meta 미달.

### 왜 폐기했는가

실험 A6와 동일 원인. HistGBT 자체가 30K val 데이터에 과적합. weight 파라미터 조정으로 근본 원인 해결 불가.

### 교훈

> "근본 원인이 모델 복잡도면 hyperparameter 튜닝으로 해결 불가. 모델 자체를 교체해야."

### Code reference

- 제출 파일: `log/submission_recal_r2_w114_t460.csv`, `log/submission_recal_r3_w114_t435.csv`

---

## 실험 A8: Threshold Recalibration Grid (132조합)

### 무엇을 했나

HistGBT meta 실패 후, weight × threshold 복합 탐색으로 전환.

```python
WEIGHT_OPTIONS  = [1.00, 1.05, 1.10, 1.15]     # 4개
THRESHOLD_GRID  = np.arange(0.42, 0.585, 0.005) # 33개
# 4 × 33 = 132 조합 grid search
```

20% val split에서 (weight, threshold) MCC 최대 조합 탐색.

### 결과 (log/threshold_search.csv 인용)

threshold_search.csv 상위 행 (val MCC 기준):

| threshold | val_mcc | precision | recall | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|
| 0.495 | **0.7457** | 0.8749 | 0.8694 | 13011 | 1861 | 1954 | 13173 |
| 0.500 | 0.7456 | 0.8774 | 0.8659 | 12958 | 1810 | 2007 | 13224 |
| 0.490 | 0.7450 | 0.8724 | 0.8719 | 13048 | 1908 | 1917 | 13126 |

recalibration grid (weight × threshold 132조합) 결과 상위:

| Rank | Setting | Val MCC | LB MCC |
|---|---|---|---|
| 1 | w=1.15, t=0.455 | 0.7745 | 0.753 |
| 2 | w=1.15, t=0.460 | 0.7740 | 0.752 |
| 3 | w=1.15, t=0.435 | 0.7735 | 0.752 |

- 전 조합이 LogReg meta LB 0.757 미달
- Val에서 최적이었던 조합이 LB에서 무효 → **val 과적합 추가 증거**

### 왜 폐기했는가

- **132조합 동시 탐색 → val 과적합 위험 극대화**
- threshold grid만으로는 모델 분포 자체를 변경 불가
- 본질적 모델 개선 없이 val 점수만 높임

### 교훈

> "Val grid search로 hyperparameter 2개 동시 튜닝 시 과적합 위험 급증."
> "CV std도 함께 확인하라."

### Code reference

- 임계값 탐색 로그: `log/threshold_search.csv` (52행 × 11열, threshold 0.40~0.65)
- 제출 파일: `log/submission_recal_r{1,2,3}_*.csv` (3개)

---

## 실험 A9: Threshold Tuning 단독 (LogReg meta)

### 무엇을 했나

LB 0.750(4-model ensemble) 시점 진단: FP > FN → threshold 상승으로 FP 감소 시도.

```python
# LogReg meta 모델에서 단독 sweep
# threshold_search.csv: 0.40~0.65, 52개 값
# 제출: t=0.500 / t=0.505 / t=0.510
```

제출 파일: `log/submission_t0.500_fixed.csv`, `log/submission_t0.505_fixed.csv`, `log/submission_t0.510_fixed.csv`

### 결과

| threshold | LB MCC |
|---|---|
| 0.500 | **0.750** (기준) |
| 0.505 | 0.750 (변화 없음) |
| 0.510 | **0.749** (하락!) |

threshold_search.csv 기준 val MCC peak: t=0.495에서 **0.7457** (t=0.500의 0.7456 대비 미미).

### 왜 폐기했는가

**모델이 0.5에서 이미 최적 캘리브레이션 상태.**

- FP=3,381 / FN=2,882 비율이 거의 1:1 → 평형 상태
- threshold ↑ → FP 감소, 동시에 TP 동일 비율 감소 → **zero-sum trade-off**
- threshold 이동 = 동일 정보로 점수 재분배 (정보량 추가 없음)

### 교훈

> "Threshold tuning은 모델 분포를 변경하지 않음. 본질적 개선 아님."
> "FP > FN 패턴이 모든 threshold에서 zero-sum이면 모델 표현력 한계."

대체 방안: ComplementNB 추가 → Augmented Stacking → 정보 차원 추가.

### Code reference

- sweep 결과: `log/threshold_search.csv` (52행, t=0.400~0.650)
- 제출 파일: `log/submission_t0.500_fixed.csv`, `log/submission_t0.505_fixed.csv`, `log/submission_t0.510_fixed.csv`

---

## 핵심 교훈 정리

### 일반화 원칙 (T1)

1. **CV-LB gap이 모델 선택의 1순위 지표.** Val 점수 단독 신뢰 금지.
2. **모델 복잡도 ≤ 데이터 크기.** 30K val에 GBT = 과적합.
3. **Zero-sum trade-off는 본질적 한계 신호.** 우회 불가.

### 엔지니어링 원칙 (T2)

4. **Solver 간 hyperparameter 등가 환산 확인.** SGD alpha = 1/(C·N).
5. **옵션 상호작용 검증.** `average=True`가 sparsity 무력화하는 함정 존재.
6. **블랙리스트 > 화이트리스트.** 화이트리스트는 어미/시제 정보 손실.

### 앙상블 원칙 (T3)

7. **다양성은 모델 클래스 차이에서 나옴.** 같은 알고리즘의 변형은 시너지 없음.
8. **메타 학습 데이터에 맞는 메타 학습기 선택.** 작은 데이터엔 단순 모델.

### 평가 원칙 (T4)

9. **Threshold tuning은 보너스. 정보가 충분하면 0.5에서 최적.**
10. **Recall/Precision은 trade-off.** 한쪽만 개선 시 다른 쪽 손해.

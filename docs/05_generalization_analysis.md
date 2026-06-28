# 일반화 성능 분석

본 프로젝트에서 모델 선택·폐기 결정을 지배한 일반화 원칙과 실증 데이터의 종합 분석.

---

## 1. Generalization이란 무엇인가

**정의:** 학습 데이터(train)에서 측정한 성능이 아닌, 학습 중 한 번도 본 적 없는 데이터(unseen test)에서의 성능.

본 프로젝트 맥락:
- **Val MCC**: 80/20 split의 20% holdout 또는 5-fold CV 검증 점수
- **LB MCC**: Kaggle-style 채점 서버(공개 테스트셋 49,997행) 점수
- **운영 기준**: `min(Group1 MCC, Group2 MCC)` — 가이드 §9 명시. 두 그룹 중 낮은 쪽이 최종 점수.

Val MCC가 높아도 LB MCC가 낮으면 일반화 실패. 이 간격을 **CV-LB gap**으로 정량화.

---

## 2. CV-LB Gap — 모델 선택의 1순위 지표

### 정의

```
CV-LB gap = Val MCC − LB MCC
```

- gap > 0: Val이 LB보다 높음 → val 과적합 의심
- gap < 0: Val이 LB보다 낮음 → val 세트가 어렵거나 LB에서 우연히 향상
- gap ≈ 0, 소폭 양수: 일반화 양호

### 실증 비교

| 실험 | Val MCC | LB MCC | CV-LB Gap | 채택 | 판정 |
|---|---|---|---|---|---|
| LogReg L2 meta | 0.7456 | **0.7570** | **+0.011** | ✓ | 일반화 양호 |
| HistGBT meta (1.25) | 0.7748 | 0.7500 | −0.025 | ✗ | 과적합 |
| HistGBT meta (1.15) | 0.7745 | 0.7530 | −0.022 | ✗ | 과적합 |
| Threshold grid (best) | 0.7745 | 0.753 | −0.022 | ✗ | val 과적합 |

참조: `docs/04_experiments_appendix.md §실험 A6, A7, A8`

**결론:** gap 부호와 크기가 선택 기준. Val 점수 단독 신뢰 금지.

LogReg meta는 Val에서 HistGBT 대비 −0.029 열위였으나 LB에서 +0.007 우위 — **진짜 성능은 LB가 결정.**

---

## 3. Bias-Variance Trade-off (모델 복잡도 vs 데이터 크기)

### 메타 학습 환경

- 데이터 규모: **30,000 샘플 × 16 피처**
  - 4개 base 모델 확률 + 12개 hand-crafted 피처
- 16 피처는 저차원 → 메타 학습기 복잡도를 낮게 유지해야

### HistGBT의 over-capacity 분석

```
max_depth = 6
→ 최대 leaf 수 = 2^6 = 64
→ 샘플/leaf = 30,000 / 64 ≈ 468

468 샘플/leaf 수준에서 GBT가 leaf-specific noise를 학습하기 시작.
```

- **High variance**: HistGBT가 val 세트의 샘플별 패턴까지 기억
- 처음 보는 LB 데이터에 해당 noise 패턴 없음 → 성능 폭락

### LogReg의 under-capacity 우위

```
파라미터 수 = 16 (weight) + 1 (bias) = 17개
30,000 / 17 ≈ 1,765 샘플/파라미터

외울 capacity 자체 부족 → 일반 패턴만 학습 → 일반화 우수
```

- **Low variance**: 16개 weight로 sample-specific noise 학습 불가
- 오히려 이 제약이 일반화 성능을 보호

### 일반 원칙

> 모델 복잡도 ≤ 데이터 크기에 비례하여 설정.
> 30K × 16 피처 환경 → LogReg가 GBT보다 적합한 메타 학습기.

참조: `quiz_prep.md §Q6, Q23`

---

## 4. Validation 전략 진화

### 단계별 변화

| 단계 | 검증 방법 | 이유 |
|---|---|---|
| Stage 0 (baseline) | 5-fold CV | 초기 모델 신뢰성 검증. 충분한 시간 확보. |
| Stage 1 (LogReg single) | 5-fold CV → LB 확인 | CV-LB gap 0.007 확인 → single split 신뢰성 근거 확보 |
| Stage 5+ (3-View 이후) | 80/20 single split | Colab 12시간 제한. 5-fold = 7.5시간 불가. 30K holdout으로 충분. |
| Drill-06 (가이드 해제) | 5-fold stratified OOF | GPU 환경, 시간 여유. OOF prediction으로 누수 원천 차단. |

### 80/20 single split의 타당성 근거

- Stage 1에서 5-fold CV val_mcc=0.7402 → LB=0.733 (gap=0.007 소폭 양수)
- 이 gap 수준이 단일 split에서도 신뢰성 확보됨을 실증
- 30,000 샘플 holdout은 LogReg 16 파라미터 검증에 충분 (1,000+ 샘플로도 검증 가능)

참조: `quiz_prep.md §Q10`

### Leakage 방지 메커니즘

**2-Phase 학습 구조:**

```
Phase 1: 80% train → base 모델 학습
          → 20% val 예측 (OOF prediction)
          → 16d 메타 피처 생성
          → 메타 LogReg 학습 (val 세트로)

Phase 2: 100% train → base 모델 재학습
          → test 예측 생성 (처음 보는 데이터)
```

핵심: 메타 학습기의 입력이 base 모델이 학습 중 **본 적 없는** 데이터의 예측이어야 unbiased.
Phase 1 학습 데이터로 예측하면 base 모델이 정답을 외운 상태 → 비현실적 확률 → 메타 학습기가 가짜 신호 학습.

참조: `quiz_prep.md §Q2`

**Drill-06 강화 (5-fold OOF):**

```
각 fold에서: 4/5 학습 → 1/5 예측
5개 OOF 예측 결합 → 전체 train 커버
→ 모든 샘플에 대해 unseen 예측 확보
```

---

## 5. MCC가 평가 지표인 이유

### 수식

```
MCC = (TP × TN − FP × FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

모든 confusion matrix의 4개 셀(TP/TN/FP/FN)을 분자·분모 양쪽에서 고려.

### Accuracy 함정 시나리오

본 데이터셋은 POSITIVE:NEGATIVE = **50:50 균형** (public_train.csv 149,995행).

| 예측 전략 | TP | TN | FP | FN | Accuracy | MCC |
|---|---|---|---|---|---|---|
| 전부 POSITIVE 예측 | 25,000 | 0 | 25,000 | 0 | **50%** | **0** |
| 균형 잡힌 좋은 예측 | 23,000 | 23,000 | 2,000 | 2,000 | 92% | 0.84 |

Accuracy 50%도 MCC=0으로 "무작위 추측과 동등"임을 정확히 표현.

### MCC 값 범위

- **+1**: 완벽한 예측
- **0**: 무작위 추측
- **−1**: 완전 반대 예측 (체계적 오류)

### Φ-coefficient 동치성

MCC는 통계학의 **Φ(파이)-coefficient** (두 이진 변수 간 상관계수)와 수학적으로 동일.
50:50 균형에서도 Accuracy보다 엄격한 이유: 분모가 클래스 분포 불균형을 자동 보정.

참조: `quiz_prep.md §Q15`, 가이드 §9

---

## 6. Threshold Tuning의 실패가 알려준 것

### 0.5에서의 캘리브레이션

LogReg는 sigmoid 출력으로 **확률을 직접 모델링** → 0.5가 최적 결정 경계인 상태가 정상.

threshold_search.csv (52개 값, t=0.400~0.650) 분석:

| threshold | val_mcc | fp_minus_fn |
|---|---|---|
| 0.490 | 0.7450 | −9 |
| **0.495** | **0.7457** | −93 |
| **0.500** | 0.7456 | −197 |
| 0.505 | 0.7450 | −287 |

t=0.495에서 val MCC 최고(0.7457)이나 t=0.500 대비 **차이 0.0001** — 무시 가능한 수준.
t=0.510 이상에서 LB가 실제 하락 (0.749).

### Zero-sum trade-off 구조

```
threshold ↑ : FP 감소 → Precision 향상
              동시에 TP 감소 → Recall 하락
              MCC 변화 없음 또는 하락 → zero-sum
```

FP=3,381 / FN=2,882 비율이 거의 1:1 → 평형 상태. threshold 이동이 한쪽을 개선하면 다른 쪽이 동일 비율로 악화.

### Val grid 132조합의 과적합

(weight × threshold) 132조합 동시 탐색:
- val에서 최적 조합 (w=1.15, t=0.455) → LB 0.753 (LogReg meta 0.757 미달)
- **val이 작을수록, 탐색 차원이 많을수록 과적합 위험 증가**

**핵심 메시지:** Threshold tuning은 모델의 **분포 자체를 변경하지 않음**. 본질적 성능 개선 불가. 기껏해야 소폭 보너스.

참조: `docs/04_experiments_appendix.md §실험 A8, A9`, `quiz_prep.md §Q21`

---

## 7. 실증 데이터 종합 표

| 실험명 | Val MCC | LB MCC | Gap | 채택 | 일반화 평가 |
|---|---|---|---|---|---|
| Baseline LogReg(saga) | 0.7402 (CV) | 0.733 | +0.007 | ✓ | 양호 — gap 소폭 양수 |
| 3-View Ensemble | — | 0.750 | — | ✓ | 양호 — Val 미측정, LB 향상 |
| 4-Model (+ComplementNB) | — | 0.753 | — | ✓ | 양호 — 다양성 기반 안정 향상 |
| Augmented Stacking (LogReg meta) | 0.7456 | **0.757** | +0.011 | ✓  | 최우수 — gap 양수, LB 최고 |
| HistGBT meta (1.25) | 0.7748 | 0.750 | **−0.025** | ✗ | 과적합 — gap 음수 |
| HistGBT meta (1.15) | 0.7745 | 0.753 | **−0.022** | ✗ | 과적합 — gap 음수 |
| Threshold grid (best) | 0.7745 | 0.753 | −0.022 | ✗ | val 과적합 |
| Threshold 단독 (t=0.510) | — | 0.749 | — | ✗ | zero-sum 확인 |

**종합:** Val MCC만으로 줄 세우면 HistGBT meta(0.7748)가 1위이나 실제 LB에서 꼴찌. CV-LB gap이 진짜 지표.

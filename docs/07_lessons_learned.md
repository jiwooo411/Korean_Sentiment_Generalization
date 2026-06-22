# 10대 교훈 (Lessons Learned)

9개 폐기 실험과 5개 채택 실험에서 귀납한 재현 가능한 원칙 10개 — T1>T2>T3>T4 우선순위 계층 기반.

---

## 교훈 카드

### 1. CV-LB gap이 모델 선택의 1순위 지표

**원칙**: Val MCC가 높아도 CV-LB gap이 음수면 과적합이므로 즉시 폐기.

**근거 실험**: [Stage 8] HistGBT meta — Val 0.7748 vs LB 0.750 (gap −0.025)

**한 줄 메시지**: Val 점수 단독 신뢰 금지. gap 부호가 모든 것을 결정.

---

### 2. 모델 복잡도 ≤ 데이터 크기

**원칙**: 메타 학습 환경 30K × 16 피처에서는 LogReg(16 파라미터)가 HistGBT(64 leaf)보다 일반화 우수.

**근거 실험**: [Stage 8] HistGBT max_depth=6 → 30K/64 ≈ 468 샘플/leaf → noise 학습 시작

**한 줄 메시지**: 강력한 모델이 항상 좋지 않음. 데이터 크기에 비례한 복잡도 선택.

---

### 3. Zero-sum trade-off는 model expressivity ceiling 신호

**원칙**: threshold 이동 시 FP↓=TP↓이면 모델 분포 자체가 한계에 도달한 것이므로 우회 불가.

**근거 실험**: [Stage 10] threshold 0.500→0.505→0.510 sweep — LB 0.750→0.750→0.749

**한 줄 메시지**: zero-sum이면 threshold 조정이 아닌 모델/피처 자체를 개선해야.

---

### 4. Solver 등가식: SGD α = 1/(C·N)

**원칙**: SGDClassifier로 LogisticRegression을 대체할 때 반드시 등가 alpha 환산 확인.

**근거 실험**: [Appendix 실험 A2] alpha=1e-4가 등가값 4e-6 대비 25배 과정규화 → val_mcc=0.6450

**한 줄 메시지**: 솔버 교체 = hyperparameter 재설계. 직접 이식은 underfit 보장.

---

### 5. 옵션 상호작용 검증 (average=True × ElasticNet)

**원칙**: 새 옵션 추가 전 기존 옵션과의 상호작용 전수 확인. 특히 정규화 관련 옵션.

**근거 실험**: [Appendix 실험 A3] `average=True` × ElasticNet → 전 12조합 sparsity=0% (L1 무력화)

**한 줄 메시지**: Polyak averaging은 L1 weight를 0으로 만들지 못하게 한다.

---

### 6. Blacklist > Whitelist (전처리)

**원칙**: POS 화이트리스트는 어미(EC/EF/ETM) 등 시제·존칭 정보를 제거하므로 블랙리스트 방식이 일반화에 유리.

**근거 실험**: [Appendix 실험 A1] POS 9종 화이트리스트 → CV 0.6915 정체 vs 블랙리스트 → LB 0.733 달성

**한 줄 메시지**: 적극적 제거보다 최소 개입. TF-IDF의 min_df/max_df에 필터링을 위임.

---

### 7. 다양성 = 모델 클래스 차이 > hyperparameter 변형

**원칙**: 앙상블 시너지는 같은 알고리즘의 파라미터 변형이 아닌 서로 다른 모델 패러다임에서 발생.

**근거 실험**: [Appendix 실험 A4] SGD v3(average=False) — LogReg와 예측 상관 ≈0.95 → 시너지 없음 / [Stage 6] ComplementNB 추가 (Generative 패러다임) → LB +0.003

**한 줄 메시지**: 다양성 = Discriminative × Generative × 표현 차원. 솔버 변형은 다양성이 아님.

---

### 8. 메타 학습기 복잡도 ≤ 메타 데이터 크기 (T3)

**원칙**: Stacking의 메타 학습기는 메타 데이터(30K × 16) 규모에 맞게 단순하게 유지.

**근거 실험**: [Stage 7] LogReg L2 meta → LB 0.757 / [Stage 8] HistGBT meta → LB 0.750 (과적합)

**한 줄 메시지**: 메타는 단순하게. 복잡한 메타 학습기는 base 예측의 noise를 학습한다.

---

### 9. Threshold tuning은 보너스, 본질적 개선 아님

**원칙**: 잘 캘리브레이션된 모델(LogReg sigmoid)에서 threshold=0.5가 이미 MCC 최적. 이동은 zero-sum.

**근거 실험**: [Stage 10] t=0.500: LB 0.750, t=0.510: LB 0.749 (하락) / threshold_search.csv: t=0.495 val peak 0.7457 (t=0.500 대비 +0.0001 — 무시 가능)

**한 줄 메시지**: threshold tuning은 보너스 조정. 정보 추가 없이 분포만 재배분.

---

### 10. Val grid 동시 튜닝 = overfit 위험

**원칙**: (weight × threshold) 같은 2차원 grid를 같은 val 세트에서 동시 탐색하면 과적합 위험이 급증.

**근거 실험**: [Stage 9] 132조합 grid → val peak 0.7745 but LB 0.753 (LogReg meta 0.757 미달)

**한 줄 메시지**: Val grid 탐색 차원이 늘수록 val 점수는 LB와 괴리. 탐색 차원 최소화.

---

## 메타-교훈

한 학생이 sklearn-only 제약 환경에서 LB 0.733에서 출발해 0.757(Rank 5)에 도달한 과정은 "더 좋은 모델을 찾는 여정"이 아니라 **시행착오를 통해 일반화 성능의 결정 요인을 이해하는 여정**이었다. 직관("정교한 전처리가 답")은 실험 A1에서 바로 부정됐고, 속도 최적화 시도(SGD)는 세 번의 실패 후에야 "다양성은 솔버가 아닌 모델 클래스에서 나온다"는 교훈으로 전환됐다. 가장 강력한 학습은 Stage 8이었다: Val MCC가 가장 높았던 HistGBT meta를 CV-LB gap 한 줄로 폐기하고 "덜 복잡한" LogReg를 확정한 결정. 이 결정이 Rank 5를 만들었다.

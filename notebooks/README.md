# Notebooks

프로젝트 정본(canonical) 실행 코드. 노트북 2개로 Phase 1·2 전체 재현.

| 노트북 | Phase | 표현·모델 | 출력 | 실행 환경 |
|---|---|---|---|---|
| `phase1_sklearn_stacking.ipynb` | 1 | 3-View TF-IDF × (LogReg ElasticNet · Complement NB) → Augmented Stacking | Public 0.757 → Hidden 0.7522 | CPU 가능 (Colab T4 약 90분) |
| `phase2_embedding_stacking.ipynb` | 2 | TF-IDF × multilingual-e5 × KR-SBERT + 5-fold OOF stacking | Hidden 0.753970 (3/18) | GPU 필수 |

## 실행 전 준비

1. `pip install -r ../requirements.txt`
2. [`data/README.md`](../data/README.md) 안내대로 `public_train.csv` · `public_test.csv` 배치
3. `SEED=42` 고정 — fold별 vectorizer/scaler 재fit으로 leakage 차단, threshold sweep은 OOF 한정

> 대용량 산출물(`*.pkl`, 임베딩 `*.npy`, 제출 CSV)은 `.gitignore` 처리. 핵심 결과는 [`results/`](../results/) 참조.

# Results

전체 실험 로그(`log/`, 1.5 GB)에서 **재현·검증에 필요한 핵심만 추린 폴더**.

## 파일

| 파일 | 내용 |
|---|---|
| `experiments_master.csv` | 14개 실험 + Phase 결과 통합 마스터 로그 (stage / 표현 / 모델 / Val / Public LB / Final Hidden / Δ / 채택여부 / 사유) |
| `ensemble_3view_results.csv` | 3-View 앙상블 가중 조합별 Val MCC (best 0.7456) |

## 최종 성과 요약

| Phase | 트랙 | Public LB | Final Hidden MCC | 순위 |
|---|---|---|---|---|
| 1 (Drill-05) | sklearn-only stacking | 0.757 (public 5위) | **0.7522** | **9 / 19** |
| 2 (Drill-06) | TF-IDF × e5 × KR-SBERT | — | **0.753970** | **3 / 18** |

- Phase 2 개선폭: 동일 Hidden 테스트에서 sparse 0.665278 → dense 0.753970 (**+0.088692**).
- 대용량 제출 CSV·`.pkl`·임베딩 `.npy`는 `.gitignore` 처리 (로컬 `log/`에 보존).

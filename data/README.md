# Data

원본 데이터는 용량·재배포 제약으로 저장소에 포함하지 않음 (`.gitignore` 처리).

## 출처

- **NSMC 계열** (Naver Sentiment Movie Corpus) 한국어 영화 리뷰 이진 분류 데이터.
- 원본: <https://github.com/e9t/nsmc>

## 기대 파일 구조

노트북 실행 전 아래 두 파일을 이 폴더(또는 노트북 `DATA_DIR`)에 위치:

| 파일 | 행 수 | 컬럼 |
|---|---|---|
| `public_train.csv` | 149,995 | `row_id`, `text`, `label` |
| `public_test.csv` | 49,997 | `row_id`, `text` |

- `label` ∈ {`POSITIVE`, `NEGATIVE`} — 학습 데이터 클래스 균형 ≈ 50.1% / 49.9% (NEGATIVE 75,170 / POSITIVE 74,825).

## 노트북 경로 설정

```python
from pathlib import Path
DATA_DIR = Path("data")  # 또는 Colab Drive 경로로 교체
train = pd.read_csv(DATA_DIR / "public_train.csv")
test  = pd.read_csv(DATA_DIR / "public_test.csv")
```

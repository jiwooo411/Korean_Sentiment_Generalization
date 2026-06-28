# Contributing

개인 저장소이지만 협업 저장소 기준의 Git 워크플로우를 적용한다. AI 에이전트(Claude · Codex 등)를 사용한 작업에도 동일하게 적용한다.

## Workflow

`main` pull → 작업 브랜치 생성 → commit → push → Pull Request → 리뷰 → merge

## Branch naming

`<type>/<short-topic>` 형식. 예: `docs/readme-polish`, `feat/oof-stacking`.
`claude/*`, `codex/*` 등 임의 브랜치명은 금지한다.

## Commit message

[Conventional Commits](https://www.conventionalcommits.org) 규칙을 따른다.

| Type | 용도 |
|---|---|
| feat | 새로운 기능 추가 |
| fix | 버그 수정 |
| docs | 문서 변경 |
| refactor | 동작 변경 없는 코드 개선 |
| chore | 빌드·설정 등 기타 변경 |

## Pull Request

무엇을(what), 왜(why), 리뷰 포인트만 간단히 작성한다. Merge는 저장소 소유자 확인 후 수행한다.

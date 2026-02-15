#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "[ERROR] origin remote가 없습니다. 먼저 원격 저장소를 연결하세요."
  echo "예: git remote add origin <repo-url>"
  exit 1
fi

echo "[1/5] origin 최신 내용 가져오기"
git fetch origin

echo "[2/5] 현재 브랜치: ${CURRENT_BRANCH}"

echo "[3/5] origin/${BASE_BRANCH} 병합 시도"
set +e
git merge --no-ff "origin/${BASE_BRANCH}"
MERGE_CODE=$?
set -e

if [ $MERGE_CODE -ne 0 ]; then
  echo ""
  echo "[충돌 발생] 아래 파일들을 열어 <<<<<<< / ======= / >>>>>>> 마커를 정리하세요."
  git diff --name-only --diff-filter=U
  echo ""
  echo "정리 후 실행:"
  echo "  git add <충돌해결된 파일들>"
  echo "  git commit -m \"Resolve merge conflicts with ${BASE_BRANCH}\""
  echo "  git push"
  exit 1
fi

echo "[4/5] 병합 완료. 원격으로 push"
git push

echo "[5/5] 완료: GitHub PR 페이지 새로고침"

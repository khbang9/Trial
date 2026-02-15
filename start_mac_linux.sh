#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3가 필요합니다. 설치 후 다시 실행해주세요."
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --disable-pip-version-check -r requirements.txt

echo "브라우저에서 http://127.0.0.1:8000 접속"
python -m uvicorn app:app --host 127.0.0.1 --port 8000

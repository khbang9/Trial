@echo off
setlocal
cd /d %~dp0

echo [1/4] Python 확인 중...
where python >nul 2>nul
if errorlevel 1 (
  echo Python이 설치되어 있지 않습니다. https://www.python.org/downloads/ 에서 설치 후 다시 실행하세요.
  pause
  exit /b 1
)

echo [2/4] 가상환경 준비 중...
if not exist .venv (
  python -m venv .venv
)

echo [3/4] 패키지 설치/업데이트 중...
call .venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo 패키지 설치에 실패했습니다. 네트워크/프록시 환경을 확인하세요.
  pause
  exit /b 1
)

echo [4/4] 서버 실행 + 브라우저 열기
start "" http://127.0.0.1:8000
call .venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

pause

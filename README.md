# Screen Change Watcher + Auto Clicker

드래그로 지정한 화면 영역을 실시간 감시하고, 이미지 변화가 임계값을 넘으면 미리 설정한 여러 클릭 액션을 순차 실행하는 프로그램입니다.

## 기능
- 모니터 스크린샷 위에서 **드래그로 모니터링 영역 선택**
- 이미지 차이(Mean Absolute Difference) 기반 **실시간 변화 감지**
- 클릭 좌표를 여러 개 등록 가능
- 각 클릭 액션마다 클릭 횟수 / 클릭 간 인터벌 / 버튼(left/right/middle) 설정
- 변화 감지 후 쿨다운 시간 지원

## 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속.

## 사용법
1. 우측 캔버스에서 감시할 영역을 드래그로 선택
2. 좌측 패널에서 클릭 액션(좌표/횟수/인터벌/버튼) 추가
3. Threshold, Polling, Cooldown 값 조정
4. **모니터링 시작** 클릭
5. 영역 이미지가 달라지면 액션 목록이 순차 실행

## 주의
- OS 접근 권한(스크린 캡처/마우스 제어)이 필요합니다.
- `pyautogui.FAILSAFE=True`가 켜져 있어, 마우스를 화면 좌상단으로 이동하면 긴급 중단할 수 있습니다.

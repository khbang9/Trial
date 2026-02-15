# 화면 변화 감지 자동 클릭기

초보자 기준으로 최대한 단순하게 만들었습니다.
핵심은 **웹페이지는 설정 메뉴(플로팅 패널)**이고, 감지 영역/클릭 위치는 **실제 데스크톱 오버레이**에서 직접 선택하는 방식입니다.

## 가장 쉬운 실행 방법 (Windows)
1. `start_windows.bat` 더블클릭
2. 브라우저가 `http://127.0.0.1:8000`로 자동 열림
3. **영역 설정하기 (플로팅)** 버튼 클릭 → 실제 화면에서 드래그
4. **클릭 위치 화면에서 선택** 버튼 클릭 → 실제 화면에서 1번 클릭
5. 클릭 횟수/인터벌/버튼 정하고 `액션 추가`
6. `모니터링 시작`

## macOS / Linux
```bash
./start_mac_linux.sh
```

## 왜 127.0.0.1 인가요?
- `127.0.0.1`은 "내 컴퓨터 안에서만 열리는 로컬 주소"입니다.
- 외부 서버가 아니라 내 PC에서 직접 실행 중이라는 뜻이라 정상입니다.

## PR 충돌(conflict) 뜰 때 빠른 해결
GitHub에 `This branch has conflicts that must be resolved`가 뜨면, 아래 순서대로 하면 됩니다.

```bash
git checkout work
# 원격이 이미 연결된 경우
# git fetch origin
# git merge origin/main
```

- 충돌 파일(`README.md`, `app.py`, `static/index.html`)에서 `<<<<<<<`, `=======`, `>>>>>>>` 구간을 정리합니다.
- 정리 후:

```bash
git add README.md app.py static/index.html
git commit -m "Resolve merge conflicts"
git push
```

### 이 프로젝트 기준 권장 원칙
- `app.py`: API 엔드포인트는 둘 다 유지하되, 중복 함수는 1개로 정리
- `static/index.html`: 최신 UX(플로팅 패널 + 오버레이 선택 버튼) 유지
- `README.md`: 실행 방법은 초보자 버전(더블클릭/스크립트) 우선


## PR 충돌을 진짜로 바로 푸는 방법(추천)
아래 스크립트를 추가해두었습니다. GitHub에서 충돌 경고가 떠도 이걸로 한 번에 병합 시도할 수 있습니다.

```bash
./resolve_conflicts.sh main
```

- 충돌이 없으면 자동으로 `push`까지 진행됩니다.
- 충돌이 있으면 충돌 파일 목록을 보여주고, 정리 후 `add/commit/push`만 하면 됩니다.
- 기본 브랜치가 `master`라면 `./resolve_conflicts.sh master`로 실행하세요.

## 주의사항
- 오버레이 선택 기능은 OS GUI 권한이 필요합니다.
- 원격/헤드리스 환경에서는 오버레이가 뜨지 않을 수 있습니다.
- 마우스를 화면 왼쪽 위로 보내면 fail-safe로 긴급중단됩니다.

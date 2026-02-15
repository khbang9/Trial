from __future__ import annotations

import io
import queue
import threading
import time
from typing import Dict, List, Literal, Optional

import mss
import numpy as np
import pyautogui
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from pynput import keyboard

pyautogui.FAILSAFE = True

app = FastAPI(title="Screen Change Watcher")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MonitorRegion(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Point(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class ClickAction(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    clicks: int = Field(default=1, ge=1, le=20)
    interval_ms: int = Field(default=80, ge=0, le=10_000)
    button: Literal["left", "middle", "right"] = "left"
    label: str = Field(default="Action")


class MonitorRule(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)
    region: MonitorRegion
    threshold: float = Field(default=12.0, ge=0.1, le=255.0)
    cooldown_ms: int = Field(default=1000, ge=0, le=60_000)
    actions: List[ClickAction]

    @model_validator(mode="after")
    def _check_actions(self) -> "MonitorRule":
        if not self.actions:
            raise ValueError("At least one action is required for each rule")
        return self


class MonitorConfig(BaseModel):
    poll_interval_ms: int = Field(default=250, ge=50, le=5_000)
    rules: List[MonitorRule]

    @model_validator(mode="after")
    def _check_rules(self) -> "MonitorConfig":
        if not self.rules:
            raise ValueError("At least one monitoring rule is required")
        return self


class HotkeyConfig(BaseModel):
    start_hotkey: str = Field(default="Ctrl+Alt+S")
    stop_hotkey: str = Field(default="Ctrl+Alt+X")


class StatusOverlay:
    def __init__(self) -> None:
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def _ensure(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:  # pragma: no cover
        try:
            import tkinter as tk
        except Exception:
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.85)
        root.geometry("290x42+14+14")
        root.configure(bg="#0f172a")
        label = tk.Label(root, text="", fg="#e2e8f0", bg="#0f172a", font=("Arial", 10, "bold"))
        label.pack(fill="both", expand=True)
        root.withdraw()

        def pump() -> None:
            while not self._q.empty():
                mode, text = self._q.get_nowait()
                if mode == "show":
                    label.configure(text=text, bg="#0f172a")
                    root.configure(bg="#0f172a")
                    root.deiconify()
                elif mode == "event":
                    label.configure(text=text, bg="#14532d")
                    root.configure(bg="#14532d")
                    root.deiconify()
                    root.after(1800, lambda: self._q.put(("hide", "")))
                elif mode == "hide":
                    root.withdraw()
            root.after(120, pump)

        root.after(120, pump)
        root.mainloop()

    def show(self, text: str) -> None:
        self._ensure()
        self._q.put(("show", text))

    def event(self, text: str) -> None:
        self._ensure()
        self._q.put(("event", text))

    def hide(self) -> None:
        self._q.put(("hide", ""))


class MonitorState:
    def __init__(self, overlay: StatusOverlay) -> None:
        self.config: Optional[MonitorConfig] = None
        self.hotkeys = HotkeyConfig()
        self.running = False
        self.last_diff_by_rule: Dict[str, float] = {}
        self.last_message = "대기중"
        self.last_triggered_rule: Optional[str] = None
        self.completed_count: int = 0
        self._overlay = overlay
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "last_message": self.last_message,
                "last_triggered_rule": self.last_triggered_rule,
                "completed_count": self.completed_count,
                "last_diff_by_rule": self.last_diff_by_rule,
                "config": self.config.model_dump() if self.config else None,
                "hotkeys": self.hotkeys.model_dump(),
            }

    def set_config(self, config: MonitorConfig) -> None:
        with self._lock:
            self.config = config
            self.last_message = f"설정 저장 완료 ({len(config.rules)}개 구역)"
            self.last_diff_by_rule = {r.id: 0.0 for r in config.rules}

    @staticmethod
    def _capture_region(region: MonitorRegion) -> np.ndarray:
        monitor = {"left": region.x, "top": region.y, "width": region.width, "height": region.height}
        with mss.mss() as sct:
            shot = sct.grab(monitor)
            arr = np.array(shot, dtype=np.uint8)
            return arr[:, :, :3]

    @staticmethod
    def _run_actions(actions: List[ClickAction]) -> None:
        for act in actions:
            pyautogui.click(x=act.x, y=act.y, clicks=act.clicks, interval=act.interval_ms / 1000, button=act.button)

    def start(self, source: str = "버튼") -> None:
        with self._lock:
            cfg = self.config
        if cfg is None:
            raise RuntimeError("모니터링 설정이 없습니다. 먼저 구역 규칙을 저장하세요.")

        self.stop(show_message=False)
        with self._lock:
            self.running = True
            self.last_message = f"모니터링 시작 ({source})"
            self.last_triggered_rule = None
            self._stop.clear()
            self.completed_count = 0
        self._overlay.show(f"모니터링중 · 중지:{self.hotkeys.stop_hotkey}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, show_message: bool = True, source: str = "중지") -> None:
        t = self._thread
        self._stop.set()
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2)
        with self._lock:
            self.running = False
            if show_message:
                self.last_message = f"모니터링 해제 ({source})"
        self._overlay.hide()
        self._thread = None

    def _loop(self) -> None:
        prev_frames: Dict[str, np.ndarray] = {}
        last_trigger_at: Dict[str, float] = {}

        while not self._stop.is_set():
            with self._lock:
                cfg = self.config
            if cfg is None:
                break

            for rule in cfg.rules:
                current = self._capture_region(rule.region)
                prev = prev_frames.get(rule.id)
                if prev is not None:
                    diff = float(np.mean(np.abs(current.astype(np.float32) - prev.astype(np.float32))))
                    now = time.time()
                    trigger = diff >= rule.threshold and (now - last_trigger_at.get(rule.id, 0) >= rule.cooldown_ms / 1000)
                    with self._lock:
                        self.last_diff_by_rule[rule.id] = diff
                    if trigger:
                        self._run_actions(rule.actions)
                        last_trigger_at[rule.id] = now
                        with self._lock:
                            self.last_triggered_rule = rule.name
                            self.last_message = f"✅ 액션 수행 완료: {rule.name} · 모니터링 자동 해제"
                            self.completed_count += 1
                            self.running = False
                        self._overlay.event(f"완료: {rule.name}")
                        self._stop.set()
                        break
                prev_frames[rule.id] = current

            time.sleep(cfg.poll_interval_ms / 1000)

        with self._lock:
            self.running = False
        self._overlay.hide()


class HotkeyManager:
    def __init__(self, state: MonitorState) -> None:
        self._state = state
        self._listener: Optional[keyboard.GlobalHotKeys] = None

    @staticmethod
    def _resolve_combo(combo: str) -> str:
        map_mod = {"Ctrl": "<ctrl>", "Alt": "<alt>", "Shift": "<shift>", "Meta": "<cmd>"}
        tokens = [t.strip() for t in combo.split("+") if t.strip()]
        mods = [map_mod[t] for t in tokens if t in map_mod]
        keys = [t for t in tokens if t not in map_mod]
        if len(keys) != 1:
            raise ValueError("키 1개 + 보조키 조합만 지원")
        key = keys[0].lower()
        if key.startswith("f") and key[1:].isdigit():
            key_token = f"<{key}>"
        elif len(key) == 1:
            key_token = key
        else:
            key_token = f"<{key}>"
        candidate = "+".join(mods + [key_token])
        keyboard.HotKey.parse(candidate)
        return candidate

    def apply(self, cfg: HotkeyConfig) -> None:
        start = self._resolve_combo(cfg.start_hotkey)
        stop = self._resolve_combo(cfg.stop_hotkey)

        def safe(fn):
            def _wrap():
                try:
                    fn()
                except Exception as exc:
                    self._state.last_message = f"단축키 처리 실패: {exc}"
            return _wrap

        new_listener = keyboard.GlobalHotKeys(
            {
                start: safe(lambda: self._state.start(source=f"단축키 {cfg.start_hotkey}")),
                stop: safe(lambda: self._state.stop(source=f"단축키 {cfg.stop_hotkey}")),
            }
        )
        new_listener.start()
        if self._listener:
            self._listener.stop()
        self._listener = new_listener


overlay = StatusOverlay()
monitor_state = MonitorState(overlay)
hotkey_manager = HotkeyManager(monitor_state)
try:
    hotkey_manager.apply(monitor_state.hotkeys)
except Exception:
    monitor_state.last_message = "전역 단축키 초기화 실패 (앱은 실행됨)"


def _build_overlay_root(title: str):
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Tkinter is not available") from exc
    root = tk.Tk()
    root.title(title)
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.25)
    root.configure(bg="black")
    canvas = tk.Canvas(root, bg="black", highlightthickness=0, cursor="cross")
    canvas.pack(fill="both", expand=True)
    return tk, root, canvas


def select_region_overlay() -> MonitorRegion:
    tk, root, canvas = _build_overlay_root("영역 선택")
    result: dict[str, int] = {}
    drag: dict[str, int] = {}
    canvas.create_text(30, 30, anchor="nw", fill="#7dd3fc", font=("Arial", 16, "bold"), text="드래그로 영역 선택 · ESC 취소")
    rect_id: Optional[int] = None

    def on_press(event: tk.Event) -> None:
        nonlocal rect_id
        drag["x1"] = int(event.x)
        drag["y1"] = int(event.y)
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#22d3ee", width=3)

    def on_drag(event: tk.Event) -> None:
        if rect_id:
            canvas.coords(rect_id, drag["x1"], drag["y1"], event.x, event.y)

    def on_release(event: tk.Event) -> None:
        x1, y1 = drag.get("x1", 0), drag.get("y1", 0)
        x2, y2 = int(event.x), int(event.y)
        result.update({"x": min(x1, x2), "y": min(y1, y2), "width": max(1, abs(x2 - x1)), "height": max(1, abs(y2 - y1))})
        root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda _: root.quit())
    root.mainloop()
    root.destroy()
    if not result:
        raise RuntimeError("Selection cancelled")
    return MonitorRegion(**result)


def select_point_overlay() -> Point:
    tk, root, canvas = _build_overlay_root("클릭 위치 선택")
    result: dict[str, int] = {}
    canvas.create_text(30, 30, anchor="nw", fill="#86efac", font=("Arial", 16, "bold"), text="클릭할 위치를 1번 클릭 · ESC 취소")
    canvas.bind("<ButtonPress-1>", lambda e: (result.update({"x": int(e.x), "y": int(e.y)}), root.quit()))
    root.bind("<Escape>", lambda _: root.quit())
    root.mainloop()
    root.destroy()
    if not result:
        raise RuntimeError("Selection cancelled")
    return Point(**result)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open("static/index.html", "r", encoding="utf-8") as fp:
        return fp.read()


@app.get("/api/screenshot")
def screenshot() -> StreamingResponse:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/api/select-region")
def select_region() -> JSONResponse:
    try:
        region = select_region_overlay()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "region": region.model_dump()})


@app.post("/api/select-point")
def select_point() -> JSONResponse:
    try:
        point = select_point_overlay()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "point": point.model_dump()})


@app.post("/api/config")
def set_config(config: MonitorConfig) -> JSONResponse:
    monitor_state.set_config(config)
    return JSONResponse({"ok": True, "message": "Config saved."})


@app.post("/api/hotkeys")
def set_hotkeys(cfg: HotkeyConfig) -> JSONResponse:
    if cfg.start_hotkey == cfg.stop_hotkey:
        raise HTTPException(status_code=400, detail="시작/중지 단축키는 달라야 합니다.")
    try:
        hotkey_manager.apply(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"단축키 형식 오류: {exc}") from exc
    monitor_state.hotkeys = cfg
    return JSONResponse({"ok": True, "hotkeys": cfg.model_dump()})


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(monitor_state.status())


@app.post("/api/start")
def start(_: Optional[dict] = Body(default=None)) -> JSONResponse:
    try:
        monitor_state.start(source="버튼")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "message": "Monitoring started."})


@app.post("/api/stop")
def stop() -> JSONResponse:
    monitor_state.stop(source="버튼")
    return JSONResponse({"ok": True, "message": "Monitoring stopped."})


@app.on_event("shutdown")
def on_shutdown() -> None:
    monitor_state.stop(show_message=False)
    if hotkey_manager._listener:
        hotkey_manager._listener.stop()

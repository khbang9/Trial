from __future__ import annotations

import io
import queue
import threading
import time
from typing import Callable, List, Literal, Optional

import mss
import numpy as np
import pyautogui
from fastapi import FastAPI, HTTPException
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
    interval_ms: int = Field(default=100, ge=0, le=10_000)
    button: Literal["left", "middle", "right"] = "left"
    label: str = Field(default="Action")


class MonitorConfig(BaseModel):
    region: MonitorRegion
    threshold: float = Field(default=12.0, ge=0.1, le=255.0)
    poll_interval_ms: int = Field(default=250, ge=50, le=5_000)
    cooldown_ms: int = Field(default=1000, ge=0, le=60_000)
    actions: List[ClickAction]

    @model_validator(mode="after")
    def ensure_actions(self) -> "MonitorConfig":
        if not self.actions:
            raise ValueError("At least one click action is required")
        return self


class HotkeyConfig(BaseModel):
    start_hotkey: str = Field(default="Ctrl+Alt+S")
    stop_hotkey: str = Field(default="Ctrl+Alt+X")


class StatusOverlay:
    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1.0)

    def _run(self) -> None:  # pragma: no cover - GUI dependent
        try:
            import tkinter as tk
        except Exception:
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.86)
        root.geometry("250x40+15+15")
        root.configure(bg="#0f172a")

        label = tk.Label(root, text="", fg="#e2e8f0", bg="#0f172a", font=("Arial", 10, "bold"))
        label.pack(fill="both", expand=True)
        root.withdraw()
        self._ready.set()

        def pump() -> None:
            while not self._queue.empty():
                mode, text = self._queue.get_nowait()
                if mode == "show":
                    label.configure(text=text, bg="#0f172a")
                    root.configure(bg="#0f172a")
                    root.deiconify()
                elif mode == "event":
                    label.configure(text=text, bg="#14532d")
                    root.configure(bg="#14532d")
                    root.deiconify()
                    root.after(1800, lambda: self._queue.put(("hide", "")))
                elif mode == "hide":
                    root.withdraw()
            root.after(120, pump)

        root.after(120, pump)
        root.mainloop()

    def show_monitoring(self, text: str) -> None:
        self.start()
        self._queue.put(("show", text))

    def show_event(self, text: str) -> None:
        self.start()
        self._queue.put(("event", text))

    def hide(self) -> None:
        self._queue.put(("hide", ""))


class MonitorState:
    def __init__(self, overlay: StatusOverlay) -> None:
        self.config: Optional[MonitorConfig] = None
        self.running = False
        self.last_diff: float = 0.0
        self.last_triggered_at: Optional[float] = None
        self.last_message: str = "대기중"
        self.hotkeys = HotkeyConfig()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._overlay = overlay

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "last_diff": self.last_diff,
                "last_triggered_at": self.last_triggered_at,
                "last_message": self.last_message,
                "hotkeys": self.hotkeys.model_dump(),
                "config": self.config.model_dump() if self.config else None,
            }

    def start(self, config: Optional[MonitorConfig] = None, source: str = "버튼") -> None:
        if config is not None:
            with self._lock:
                self.config = config
        with self._lock:
            cfg = self.config
        if cfg is None:
            raise RuntimeError("모니터링 설정이 없습니다. 먼저 설정 후 시작하세요.")

        self.stop(show_message=False)
        with self._lock:
            self.last_diff = 0.0
            self.last_triggered_at = None
            self.last_message = f"모니터링중 ({source} 시작)"
            self.running = True
            self._stop_event.clear()

        self._overlay.show_monitoring(f"모니터링중 · 중지:{self.hotkeys.stop_hotkey}")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, show_message: bool = True, source: str = "중지") -> None:
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            self._stop_event.set()
            thread.join(timeout=2)
        else:
            self._stop_event.set()

        with self._lock:
            self.running = False
            if show_message:
                self.last_message = f"모니터링 해제 ({source})"
        self._thread = None
        self._overlay.hide()

    @staticmethod
    def _capture_region(region: MonitorRegion) -> np.ndarray:
        monitor = {"left": region.x, "top": region.y, "width": region.width, "height": region.height}
        with mss.mss() as sct:
            shot = sct.grab(monitor)
            arr = np.array(shot, dtype=np.uint8)
            return arr[:, :, :3]

    def _perform_actions(self, actions: List[ClickAction]) -> None:
        for action in actions:
            pyautogui.click(
                x=action.x,
                y=action.y,
                clicks=action.clicks,
                interval=action.interval_ms / 1000,
                button=action.button,
            )

    def _run_loop(self) -> None:
        prev_frame: Optional[np.ndarray] = None
        while not self._stop_event.is_set():
            with self._lock:
                config = self.config
            if not config:
                break

            current = self._capture_region(config.region)
            if prev_frame is not None:
                diff = float(np.mean(np.abs(current.astype(np.float32) - prev_frame.astype(np.float32))))
                trigger_now = False
                with self._lock:
                    self.last_diff = diff
                    if diff >= config.threshold:
                        now = time.time()
                        if not self.last_triggered_at or now - self.last_triggered_at >= (config.cooldown_ms / 1000):
                            self.last_triggered_at = now
                            trigger_now = True
                if trigger_now:
                    self._perform_actions(config.actions)
                    with self._lock:
                        self.last_message = "액션 수행됨 · 모니터링 자동 해제"
                        self.running = False
                    self._overlay.show_event("액션 수행됨 · 자동 해제")
                    self._stop_event.set()
                    break
            prev_frame = current
            time.sleep(config.poll_interval_ms / 1000)

        with self._lock:
            self.running = False


class HotkeyManager:
    def __init__(self, monitor_state: MonitorState) -> None:
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._monitor_state = monitor_state

    @staticmethod
    def _to_pynput(combo: str) -> str:
        mapping = {
            "Ctrl": "<ctrl>",
            "Alt": "<alt>",
            "Shift": "<shift>",
            "Meta": "<cmd>",
        }
        parts = combo.split("+")
        out = []
        for p in parts:
            p = p.strip()
            out.append(mapping.get(p, p.lower()))
        return "+".join(out)

    def apply(self, cfg: HotkeyConfig) -> None:
        if self._listener:
            self._listener.stop()

        start_key = self._to_pynput(cfg.start_hotkey)
        stop_key = self._to_pynput(cfg.stop_hotkey)

        self._listener = keyboard.GlobalHotKeys(
            {
                start_key: lambda: self._monitor_state.start(source=f"단축키 {cfg.start_hotkey}"),
                stop_key: lambda: self._monitor_state.stop(source=f"단축키 {cfg.stop_hotkey}"),
            }
        )
        self._listener.start()


overlay = StatusOverlay()
monitor_state = MonitorState(overlay)
hotkey_manager = HotkeyManager(monitor_state)
hotkey_manager.apply(monitor_state.hotkeys)


def _build_overlay_root(title: str):
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Tkinter is not available on this Python environment.") from exc

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

    def on_click(event: tk.Event) -> None:
        result.update({"x": int(event.x), "y": int(event.y)})
        root.quit()

    canvas.bind("<ButtonPress-1>", on_click)
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


@app.post("/api/hotkeys")
def set_hotkeys(cfg: HotkeyConfig) -> JSONResponse:
    if cfg.start_hotkey == cfg.stop_hotkey:
        raise HTTPException(status_code=400, detail="시작/중지 단축키는 달라야 합니다.")
    monitor_state.hotkeys = cfg
    hotkey_manager.apply(cfg)
    return JSONResponse({"ok": True, "hotkeys": cfg.model_dump()})


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(monitor_state.status())


@app.post("/api/start")
def start(config: MonitorConfig) -> JSONResponse:
    try:
        monitor_state.start(config=config, source="버튼")
    except Exception as exc:  # pragma: no cover
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

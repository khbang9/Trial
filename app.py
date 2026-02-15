from __future__ import annotations

import io
import threading
import time
from typing import List, Literal, Optional

import mss
import numpy as np
import pyautogui
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from PIL import Image

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


class MonitorState:
    def __init__(self) -> None:
        self.config: Optional[MonitorConfig] = None
        self.running = False
        self.last_diff: float = 0.0
        self.last_triggered_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "last_diff": self.last_diff,
                "last_triggered_at": self.last_triggered_at,
                "config": self.config.model_dump() if self.config else None,
            }

    def start(self, config: MonitorConfig) -> None:
        self.stop()
        with self._lock:
            self.config = config
            self.last_diff = 0.0
            self.last_triggered_at = None
            self.running = True
            self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=2)
        with self._lock:
            self.running = False
        self._thread = None

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
            prev_frame = current
            time.sleep(config.poll_interval_ms / 1000)

        with self._lock:
            self.running = False


monitor_state = MonitorState()


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


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(monitor_state.status())


@app.post("/api/start")
def start(config: MonitorConfig) -> JSONResponse:
    try:
        monitor_state.start(config)
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "message": "Monitoring started."})


@app.post("/api/stop")
def stop() -> JSONResponse:
    monitor_state.stop()
    return JSONResponse({"ok": True, "message": "Monitoring stopped."})


@app.on_event("shutdown")
def on_shutdown() -> None:
    monitor_state.stop()

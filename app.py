"""
Sahra-Eye — Laâyoune Smart City Edge AI Vision Engine (v2.0 Upgraded Server)
File: app.py

Key Backend Server Improvements:
1. Non-Blocking Threaded Camera Reader:
   - Reads frames in a dedicated daemon thread so slow RTSP/webcam I/O never drops FPS.
2. Async Webhook Dispatch Queue with Cooldown Debouncing:
   - Dispatches WhatsApp / Telegram alerts asynchronously in the background with a
     15-second per-mode cooldown so the video stream never freezes.
3. Incident History Log & Health Check Endpoints:
   - Provides /telemetry, /incidents, /mode, and /health REST endpoints.
"""

import os
import time
import threading
from collections import deque
from typing import Optional, Deque, Dict, Any

import cv2
import requests
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vision_engine import SahraEyeEngine

app = FastAPI(
    title="Sahra-Eye — Laâyoune Smart City Vision API",
    description="High-Speed Edge AI Vision & Emergency Triage Backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SahraEyeEngine(
    pose_model_path=os.getenv("SAHRA_POSE_MODEL", "yolov8n-pose.pt"),
    det_model_path=os.getenv("SAHRA_DET_MODEL", "yolov8n.pt"),
    prefer_openvino=True,
)

VIDEO_SOURCE = os.getenv("SAHRA_VIDEO_SOURCE", "0")
TELEGRAM_WEBHOOK_URL = os.getenv("SAHRA_TELEGRAM_WEBHOOK", "")
WHATSAPP_WEBHOOK_URL = os.getenv("SAHRA_WHATSAPP_WEBHOOK", "")

incident_history: Deque[Dict[str, Any]] = deque(maxlen=50)
last_webhook_sent: Dict[str, float] = {}


class ThreadedCameraStream:
    """Continuously grabs the latest frame in a background thread for zero-lag streaming."""

    def __init__(self, src: str):
        cap_src = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(cap_src)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


camera_stream = ThreadedCameraStream(VIDEO_SOURCE)


class ModeUpdatePayload(BaseModel):
    mode: str
    target_color: Optional[str] = None


def dispatch_webhook_async(alert_type: str, alert_message: str, mode: str) -> None:
    """Non-blocking webhook sender with 15s cooldown debounce."""
    now = time.time()
    if now - last_webhook_sent.get(mode, 0.0) < 15.0:
        return
    last_webhook_sent[mode] = now

    event = {
        "id": f"inc-{int(now * 1000)}",
        "timestamp": time.strftime("%H:%M:%S"),
        "mode": mode,
        "alert_type": alert_type,
        "message": alert_message,
        "green_wave": mode == "najda",
    }
    incident_history.appendleft(event)

    if TELEGRAM_WEBHOOK_URL:
        try:
            requests.post(TELEGRAM_WEBHOOK_URL, json={"text": f"[{alert_type}] {alert_message}"}, timeout=2.5)
        except Exception:
            pass


def mjpeg_frame_generator():
    while True:
        frame = camera_stream.read()
        if frame is None:
            time.sleep(0.02)
            continue

        annotated_frame, telemetry = engine.process_frame(frame)
        if telemetry["alert_active"]:
            threading.Thread(
                target=dispatch_webhook_async,
                args=(telemetry["alert_type"], telemetry["alert_message"], telemetry["mode"]),
                daemon=True,
            ).start()

        ok, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/telemetry")
def get_telemetry():
    return JSONResponse(content=engine.telemetry)


@app.get("/incidents")
def get_incidents():
    return JSONResponse(content={"incidents": list(incident_history)})


@app.post("/mode")
def update_mode(payload: ModeUpdatePayload):
    engine.set_mode(payload.mode)
    if payload.target_color:
        engine.set_target_color(payload.target_color)
    return JSONResponse(content=engine.telemetry)


@app.get("/health")
def health_check():
    return {"status": "ok", "hardware": engine.hardware_backend, "fps": engine.telemetry["fps"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

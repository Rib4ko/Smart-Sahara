export const VISION_ENGINE_PY_SOURCE = `"""
Sahra-Eye — Laâyoune Smart City Edge AI Vision Engine (v3.0 Faint-Memory & Foreshortened Pose Engine)
File: vision_engine.py

Key Backend Architecture Improvements (v3.0):
1. 4-Way Biomechanical Faint & Collapse Detection:
   - Detects both sideways falls AND foreshortened forward/backward faints (where a person
     collapses parallel to the aisle/camera and their bounding box remains narrow):
     (a) Head/Shoulder Vertical Drop vs. Standing Baseline Memory (>20% frame drop)
     (b) Vertical Height Compression (<68% of recorded standing height)
     (c) Floor-Zone Body Confines (entire body in lower half of frame: top_y > 0.48 * H)
     (d) Torso Inclination Angle (>48 deg) or Foreshortened Torso Collapse
2. Persistent Last-Position & Faint Memory Lock (TrackedPersonMemory):
   - Remembers each person's standing baseline position and last valid bounding box.
   - When a person faints and lies motionless on the floor (or if YOLO pose confidence drops
     while they are crumpled on the ground), the engine locks their fainted floor position in
     memory, draws the fall trajectory from their last standing position, and keeps the
     emergency alert active until they stand back up.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class TrackedPersonMemory:
    """
    Maintains standing baseline and persistent last-position memory for a person
    so fainted/motionless subjects on the floor are never lost when detection stops.
    """
    track_id: int
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_standing_bbox: Optional[Tuple[int, int, int, int]] = None
    peak_standing_height: float = 0.0
    baseline_top_y: float = 1.0
    top_y_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    height_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    fall_score_history: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    last_seen_time: float = field(default_factory=time.time)
    faint_locked: bool = False
    faint_locked_bbox: Optional[Tuple[int, int, int, int]] = None
    faint_start_time: Optional[float] = None
    last_downward_velocity: float = 0.0
`;

export const APP_PY_SOURCE = `"""
Sahra-Eye — Laâyoune Smart City Edge AI Vision Engine (v3.0 Upgraded Server)
File: app.py

Includes:
- ThreadedCameraStream for zero-lag video frame ingestion
- Persistent Faint Memory state reporting in /telemetry
- Non-blocking WhatsApp & Telegram emergency webhook dispatch
"""

import os
import time
import threading
from collections import deque
from typing import Optional, Deque, Dict, Any

import cv2
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vision_engine import SahraEyeEngine

app = FastAPI(title="Sahra-Eye Vision API v3.0", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = SahraEyeEngine(prefer_openvino=True)
`;

export const REQUIREMENTS_TXT_SOURCE = `# Sahra-Eye v3.0 // Laâyoune Smart City Edge AI Vision Engine
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
opencv-python>=4.10.0.84
numpy>=1.26.4
ultralytics>=8.3.0
openvino>=2024.4.0
pydantic>=2.9.0
requests>=2.32.0
`;

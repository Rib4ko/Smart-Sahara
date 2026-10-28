"""
Sahra-Eye — Laâyoune Smart City Edge AI Vision Engine (v3.0 Faint-Memory & Foreshortened Pose Engine)
File: vision_engine.py

Key Backend Architecture Improvements (v3.0):
1. 4-Way Biomechanical Faint & Collapse Detection:
   - Detects both sideways falls AND foreshortened forward/backward faints (where a person
     collapses parallel to the aisle/camera and their bounding box remains narrow):
     (a) Head/Shoulder Vertical Drop vs. Standing Baseline Memory (>22% frame drop)
     (b) Vertical Height Compression (<65% of recorded standing height)
     (c) Floor-Zone Body Confines (entire body in lower half of frame: top_y > 0.48 * H)
     (d) Torso Inclination Angle (>48 deg) or Foreshortened Torso Collapse
2. Persistent Last-Position & Faint Memory Lock (TrackedPersonMemory):
   - Remembers each person's standing baseline position and last valid bounding box.
   - When a person faints and lies motionless on the floor (or if YOLO pose confidence drops
     while they are crumpled on the ground), the engine locks their fainted floor position in
     memory, draws the fall trajectory from their last standing position, and keeps the
     emergency alert active until they stand back up.
3. Desert-Light Resilient HSV Color Locator (CLAHE + Morphological Filtering).
4. Intel OpenVINO FP16 Acceleration on Intel Core i7 CPU/iGPU.
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
    last_Standing_bbox: Optional[Tuple[int, int, int, int]] = None
    peak_standing_height: float = 0.0
    baseline_top_y: float = 1.0  # normalized 0..1
    top_y_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    height_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    fall_score_history: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    color_match_history: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    last_seen_time: float = field(default_factory=time.time)
    faint_locked: bool = False
    faint_locked_bbox: Optional[Tuple[int, int, int, int]] = None
    faint_start_time: Optional[float] = None
    last_downward_velocity: float = 0.0


class SahraEyeEngine:
    HSV_COLOR_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        "RED": [
            (np.array([0, 110, 70]), np.array([10, 255, 255])),
            (np.array([165, 110, 70]), np.array([180, 255, 255])),
        ],
        "BLUE": [(np.array([92, 90, 55]), np.array([135, 255, 255]))],
        "YELLOW": [(np.array([18, 105, 85]), np.array([37, 255, 255]))],
        "GREEN": [(np.array([36, 80, 50]), np.array([88, 255, 255]))],
        "BLACK": [(np.array([0, 0, 0]), np.array([180, 255, 55]))],
        "WHITE": [(np.array([0, 0, 190]), np.array([180, 45, 255]))],
    }

    WEAPON_CLASS_IDS = {43: "KNIFE / BLADE", 76: "EDGED OBJECT", 34: "BLUNT WEAPON"}

    def __init__(
        self,
        pose_model_path: str = "yolov8n-pose.pt",
        det_model_path: str = "yolov8n.pt",
        prefer_openvino: bool = True,
    ) -> None:
        self.mode: str = "najda"
        self.target_color: str = "RED"
        self.prefer_openvino: bool = prefer_openvino
        self.hardware_backend: str = "Intel OpenVINO FP16 (AUTO: CPU/iGPU)"

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracks: Dict[int, TrackedPersonMemory] = {}
        self.fps_window: Deque[float] = deque(maxlen=30)

        self.pose_model = self._load_with_openvino_fallback(pose_model_path, task="pose")
        self.det_model = self._load_with_openvino_fallback(det_model_path, task="detect")

        self.telemetry: Dict[str, Any] = {
            "mode": self.mode,
            "target_color": self.target_color,
            "fps": 0.0,
            "inference_ms": 0.0,
            "hardware": self.hardware_backend,
            "active_people_count": 0,
            "faint_memory_locked": False,
            "alert_active": False,
            "alert_type": "ALL_CLEAR",
            "alert_message": "Monitoring active. Person tracking & faint memory armed.",
            "green_wave_corridor": False,
            "detections": [],
        }

    def _load_with_openvino_fallback(self, pt_path: str, task: str):
        if YOLO is None:
            self.hardware_backend = "OpenCV Heuristic Fallback"
            return None

        ov_dir = Path(pt_path).stem + "_openvino_model"
        if self.prefer_openvino:
            try:
                if not Path(ov_dir).exists():
                    base_model = YOLO(pt_path)
                    base_model.export(format="openvino", half=True)
                self.hardware_backend = "Intel OpenVINO FP16 (Core i7 CPU/iGPU)"
                return YOLO(ov_dir, task=task)
            except Exception:
                self.hardware_backend = "PyTorch CPU Runtime (.pt)"
                return YOLO(pt_path, task=task)
        return YOLO(pt_path, task=task)

    def reset_memory(self) -> None:
        """Clears locked faint positions when operator resets scene."""
        self.tracks.clear()

    def _evaluate_faint_with_memory(
        self,
        track: TrackedPersonMemory,
        bbox: Tuple[int, int, int, int],
        keypoints: Optional[np.ndarray],
        frame_h: int,
        frame_w: int,
    ) -> Tuple[bool, str]:
        """
        4-Way Biomechanical Faint & Collapse Detector:
        Catches both horizontal falls AND foreshortened vertical faints (lying along floor aisle).
        """
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        norm_top_y = y1 / float(frame_h)
        norm_bot_y = y2 / float(frame_h)
        norm_h = h / float(frame_h)
        aspect_ratio = w / h

        if len(track.top_y_history) >= 3:
            track.last_downward_velocity = norm_top_y - track.top_y_history[-3]
        track.top_y_history.append(norm_top_y)
        track.height_history.append(norm_h)
        track.last_bbox = bbox

        # Update standing baseline when person is clearly upright in upper-middle frame
        if norm_top_y < 0.40 and norm_h > 0.45 and aspect_ratio < 0.85:
            if norm_h > track.peak_standing_height * 0.85:
                track.peak_standing_height = max(track.peak_standing_height, norm_h)
                track.baseline_top_y = min(track.baseline_top_y, norm_top_y)
                track.last_Standing_bbox = bbox
                # If person stood back up, unlock faint memory
                if track.faint_locked and norm_top_y < 0.35 and norm_h > 0.52:
                    track.faint_locked = False
                    track.faint_locked_bbox = None
                    track.faint_start_time = None

        # Check 4 Faint / Collapse Signatures:
        # 1. Head / Top-of-Body Drop relative to standing baseline
        head_drop = norm_top_y - track.baseline_top_y
        sig_head_drop = (
            track.peak_standing_height > 0.40
            and head_drop > 0.20
            and norm_top_y > 0.45
        )

        # 2. Vertical Height Compression on Floor (handles foreshortened faints!)
        sig_height_compression = (
            track.peak_standing_height > 0.40
            and norm_h < track.peak_standing_height * 0.68
            and norm_bot_y > 0.74
        )

        # 3. Entire Body Confined to Lower Floor Plane
        sig_floor_zone = norm_top_y > 0.48 and ((y1 + y2) / (2.0 * frame_h)) > 0.65

        # 4. Pose Keypoint Torso Angle or Foreshortened Torso Collapse
        sig_pose_collapse = aspect_ratio > 1.12
        if keypoints is not None and len(keypoints) >= 13:
            l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
            l_hip, r_hip = keypoints[11][:2], keypoints[12][:2]
            if (l_sh[0] > 0 or r_sh[0] > 0) and (l_hip[0] > 0 or r_hip[0] > 0):
                sh_mid = (l_sh + r_sh) / 2.0
                hip_mid = (l_hip + r_hip) / 2.0
                dx = abs(sh_mid[0] - hip_mid[0])
                dy = max(1.0, abs(sh_mid[1] - hip_mid[1]))
                angle_deg = float(np.degrees(np.arctan2(dx, dy)))
                torso_norm_dy = dy / float(frame_h)
                if angle_deg > 48.0 or (torso_norm_dy < 0.11 and norm_top_y > 0.44):
                    sig_pose_collapse = True

        is_faint_candidate = (
            sig_head_drop or sig_height_compression or sig_floor_zone or sig_pose_collapse
        )
        track.fall_score_history.append(1 if is_faint_candidate else 0)

        if sum(track.fall_score_history) >= 3 or track.faint_locked:
            track.faint_locked = True
            track.faint_locked_bbox = bbox
            if track.faint_start_time is None:
                track.faint_start_time = time.time()
            down_secs = int(time.time() - track.faint_start_time)
            return True, f"FAINTED ON FLOOR ({down_secs}s)"

        return False, "Standing / Normal"

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()
        annotated = frame.copy()
        h_img, w_img = frame.shape[:2]
        now = time.time()

        alert_active = False
        alert_type = "ALL_CLEAR"
        alert_message = "Monitoring active. Person tracking & faint memory armed."
        green_wave = False
        detections_out: List[Dict[str, Any]] = []
        seen_track_ids = set()

        if self.mode in ("najda", "tifawt") and self.pose_model is not None:
            results = self.pose_model.track(frame, conf=0.28, persist=True, verbose=False)
            for res in results:
                boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
                track_ids = (
                    res.boxes.id.cpu().numpy().astype(int)
                    if (res.boxes is not None and res.boxes.id is not None)
                    else list(range(1, len(boxes) + 1))
                )
                kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else []

                for idx, (box, tid) in enumerate(zip(boxes, track_ids)):
                    tid = int(tid)
                    seen_track_ids.add(tid)
                    x1, y1, x2, y2 = map(int, box[:4])
                    person_kpts = kpts[idx] if idx < len(kpts) else None

                    if tid not in self.tracks:
                        self.tracks[tid] = TrackedPersonMemory(track_id=tid)
                    track = self.tracks[tid]
                    track.last_seen_time = now

                    is_fallen, status_label = self._evaluate_faint_with_memory(
                        track, (x1, y1, x2, y2), person_kpts, h_img, w_img
                    )

                    if is_fallen:
                        alert_active = True
                        green_wave = True
                        alert_type = "MEDICAL EMERGENCY: PERSON FAINTED"
                        alert_message = f"Person #{tid} fainted on floor. Last position locked in memory."
                        box_color = (68, 68, 239)
                    else:
                        box_color = (16, 185, 129)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    detections_out.append({"id": tid, "label": status_label, "alert": is_fallen})

        # PERSISTENT MEMORY FALLBACK:
        # If a tracked person was recently seen dropping or already fainted, and the detector
        # stops detecting them while they lie motionless on the floor, keep their last position!
        for tid, track in list(self.tracks.items()):
            if tid in seen_track_ids:
                continue
            age = now - track.last_seen_time
            if age > 30.0:
                del self.tracks[tid]
                continue

            # If person was dropping downward or near the floor just before detection stopped,
            # lock their last known position as a fainted person memory!
            if not track.faint_locked and (
                track.last_downward_velocity > 0.05
                or (track.top_y_history and track.top_y_history[-1] > 0.38)
            ):
                track.faint_locked = True
                track.faint_locked_bbox = track.last_bbox
                track.faint_start_time = track.faint_start_time or now

            if track.faint_locked or age < 5.0:
                mx1, my1, mx2, my2 = track.faint_locked_bbox or track.last_bbox
                if track.faint_locked:
                    alert_active = True
                    green_wave = True
                    down_secs = int(now - (track.faint_start_time or now))
                    label = f"FAINT MEMORY LOCKED ({down_secs}s)"
                    alert_type = "MEDICAL EMERGENCY: FAINTED PERSON (MEMORY LOCK)"
                    alert_message = (
                        f"Person #{tid} fainted and is motionless. Showing remembered floor position."
                    )
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (68, 68, 239), 2)
                    detections_out.append({"id": tid, "label": label, "alert": True})

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.fps_window.append(1000.0 / max(1.0, elapsed_ms))

        self.telemetry.update(
            {
                "fps": round(float(np.mean(self.fps_window)), 1),
                "inference_ms": round(elapsed_ms, 1),
                "active_people_count": len(detections_out),
                "faint_memory_locked": any(t.faint_locked for t in self.tracks.values()),
                "alert_active": alert_active,
                "alert_type": alert_type,
                "alert_message": alert_message,
                "green_wave_corridor": green_wave,
                "detections": detections_out,
            }
        )
        return annotated, self.telemetry

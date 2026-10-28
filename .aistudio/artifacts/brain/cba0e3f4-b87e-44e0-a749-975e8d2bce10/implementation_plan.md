# Sahra-Eye: Laâyoune Smart City Edge AI Vision Engine

**Sahra-Eye** is an edge-optimized Smart City surveillance, privacy-compliant computer vision, and emergency triage platform designed for the urban ecosystem of **Laâyoune, Morocco**. It pairs a live interactive Web Command Center—capable of processing live webcam streams and uploaded test videos in real time—with a complete, production-ready Python Edge Vision Engine (`app.py` and `vision_engine.py`) built on **FastAPI**, **OpenCV**, **YOLOv8 (`yolov8n-pose.pt` & `yolov8n.pt`)**, and **Intel OpenVINO** for 30+ FPS local execution on Intel Core i7 integrated graphics.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> The following architectural and UX decisions incorporate your confirmed selections and establish the blueprint for both the interactive web console and the downloadable Python OpenVINO engine:

- **Confirmed Decision 1 — Dual Delivery (Interactive Web Console + Full Python Source Export)**: The platform delivers a functional browser-native Edge Vision Command Center alongside an integrated **Python Edge Code & Architecture Inspector** where you can view, copy, and download the complete `vision_engine.py`, `app.py`, and `requirements.txt` configured for YOLOv8 + Intel OpenVINO (`FP16`/`INT8`) local execution.
- **Confirmed Decision 2 — Live Webcam & Custom Video File Ingestion**: Operators can switch seamlessly between their **Live Browser Webcam** (`navigator.mediaDevices.getUserMedia`), **Custom Uploaded Test Videos** (MP4/WebM drag-and-drop or file picker), and built-in **Interactive Calibration Streams** to test all three operational modes live in the canvas.
- **Confirmed Decision 3 — Interactive Laâyoune Map & Boulevard Mekka Green-Wave Corridor**: A geospatial telemetry view of Laâyoune maps real city coordinates (Boulevard Mekka intersections, Place El Méchouar, ONEE Desalination Plant, Protection Civile HQ, and EST Laâyoune Campus). When **Najda-Vision** detects a fall or medical collapse, the map dynamically synchronizes traffic signal nodes along Boulevard Mekka into an animated **Green-Wave Emergency Corridor**.
- **Confirmed Decision 4 — Frugal WhatsApp/Telegram Webhook Dispatch & JSON Inspector**: Critical alerts generate structured emergency payloads for Protection Civile and municipal security via simulated WhatsApp Cloud API and Telegram Bot webhooks, complete with a live JSON payload inspector and interactive test-trigger controls.

---

### 1. Overview & Core Concept

- **What It Does**:
  1. **CNDP Law 09-08 Privacy-First Compliance Engine**: By default, every detected citizen's head/face region is masked in real time with a Gaussian blur and annotated with a green `CNDP-0908 PRIVACY` compliance label. The privacy mask automatically lifts **only** on the specific bounding box that triggers a verified critical alert (medical collapse or security threat), while keeping bystanders blurred.
  2. **Mode 1 — Najda-Vision (Fall & Medical Collapse Triage)**: Analyzes human skeletal posture (head-to-hip aspect ratio, torso inclination angle, and rapid vertical drop velocity via `yolov8n-pose.pt` keypoints) to detect pedestrian falls, heatstroke collapses, or traffic accidents. Immediately dispatches a Protection Civile alert and activates the Boulevard Mekka Green-Wave traffic corridor.
  3. **Mode 2 — Tifawt-Guard (Missing Child & Vulnerable Senior Locator)**: Performs real-time HSV color space segmentation on torso regions between shoulder and hip keypoints across six target clothing profiles (**Red, Blue, Yellow, Black, White, Green**) to locate missing children or disoriented seniors across municipal cameras without exposing biometric facial data.
  4. **Mode 3 — Critical Infrastructure Security Guard**: Protects restricted zones (such as the ONEE Desalination Plant and University Server Rooms) by combining upper-face HSV/luminance concealment analysis (detecting ski masks/balaclavas where facial skin ratio drops below threshold) with `yolov8n.pt` object detection for bladed weapons and unauthorized intrusion.
- **Target Audience / Persona**: Municipal operators at the Laâyoune Smart City Command Center, Protection Civile emergency dispatchers, and edge AI engineers deploying frugal computer vision on existing CCTV infrastructure.
- **Key Value**: Delivers sub-33ms (30+ FPS) emergency triage and missing-person search using existing municipal cameras and zero-cost messaging webhooks, while strictly enforcing Moroccan CNDP Law 09-08 citizen privacy by design.

---

### 2. User Experience & Visual Design

- **Key User Flows**:
  1. **Video Source Selection & Calibration**: The operator selects **Live Webcam**, uploads a **Custom Test Video** (MP4/WebM), or uses the **Interactive Scenario Calibrator** on the primary viewport canvas. Real-time computer vision overlays render bounding boxes, skeletal pose keypoints, torso HSV color swatches, and Gaussian face blurs directly on the 60 FPS HTML5 canvas.
  2. **Operational Mode Switching**: Using the segmented mode controller in the workspace toolbar, the operator switches between **Mode 1: Najda-Vision**, **Mode 2: Tifawt-Guard** (selecting target clothing color: Red, Blue, Yellow, Black, White, or Green), and **Mode 3: Infrastructure Guard**.
  3. **Incident Triage & Green-Wave Activation**: When a fall is detected in the video feed (or triggered via the incident test bench), the subject's bounding box switches from green (`CNDP-0908 PRIVACY` blurred) to crimson (`CRITICAL ALERT — UNMASKED FOR TRIAGE`), the **Boulevard Mekka Green-Wave Corridor** animates sequential green traffic lights from Protection Civile HQ to the incident camera node, and a WhatsApp/Telegram webhook payload appears in the dispatch inspector.
  4. **Python Edge Source Export**: Switching to the **Python OpenVINO Engine** view lets the engineer inspect and download the complete `vision_engine.py` (`SahraEyeEngine` class with OpenVINO export/fallback, HSV torso masking, pose fall heuristics, and CNDP Gaussian blur) and `app.py` (FastAPI `/video_feed` MJPEG generator and `/telemetry` endpoints).

- **Visual Identity & Theme**:
  - *Aesthetic Direction*: High-precision civic command center and hardware telemetry console—clean, utilitarian, high-contrast dark slate surfaces with zero decorative clutter.
  - *Color Palette & Mood*:
    - **60% Dominant Neutral Canvas**: Deep obsidian slate (`#0B0F17`) main viewport background and dark graphite (`#111827`) structural workspace panels.
    - **30% Structural Surfaces**: Crisp `1px solid #1E293B` hairline dividers, `#161F30` interactive control surfaces, and high-legibility cool off-white (`#F8FAFC`) and muted slate (`#94A3B8`) typography.
    - **10% Semantic Accent Budget**:
      - **Nominal / CNDP Law 09-08 Privacy Active & Green-Wave Signal**: Emerald (`#10B981`)
      - **Tifawt-Guard Target Match / Warning**: Amber Gold (`#F59E0B`)
      - **Critical Triage Alert (Fall Detected / Weapon / Concealed Face)**: Crimson (`#EF4444`)
      - **Intel OpenVINO Edge Hardware Accent**: Electric Cyan (`#06B6D4`)
  - *Typography & Hierarchy*:
    - **Display & Navigation**: `Outfit` (600 SemiBold / 700 Bold) with optical tracking compensation for dark surfaces.
    - **UI & Body Prose**: `Plus Jakarta Sans` (400 Regular / 500 Medium) for crisp scannability at 13px–15px.
    - **Telemetry, Coordinates, FPS & JSON Payloads**: `JetBrains Mono` with `font-variant-numeric: tabular-nums` so frame times, GPS coordinates, HSV thresholds, and countdown timers never jitter.
  - *Component Styling & Layout*:
    - Full-viewport 1440px desktop command layout following a strict 3-zone Top Bar Contract (`Sahra-Eye` wordmark — single-line navigation links for `Live Command`, `Laâyoune Corridor Map`, `Webhook Inspector`, `Python OpenVINO Source` — and `Export Edge Package` primary button).
    - Single-elevation panels separated by `1px` borders; metadata rendered as clean inline text with `·` separators rather than pill badges.

- **Interactive Feedback & Motion**:
  - Real-time HTML5 Canvas overlay rendering Gaussian face blur regions, skeletal pose vectors, torso HSV sampling boxes, and CNDP-0908 compliance labels at 30–60 FPS.
  - Smooth `< 150ms` transitions when switching modes, filtering target clothing colors, or inspecting webhook JSON payloads.

---

### 3. Key Product Decisions & Trade-Offs

- **Decision 1 — Real-Time Browser CV Pipeline Paired with Production Python OpenVINO Code**:
  - *Chosen Approach*: Run an authentic client-side computer vision pipeline on `<canvas>` (performing real pixel-level HSV color segmentation on uploaded videos/webcam feeds, motion/posture bounding box analysis, and real Gaussian pixel blurring on head regions) while providing the complete, downloadable Python `vision_engine.py` and `app.py` engineered for `ultralytics` YOLOv8 + Intel OpenVINO (`yolov8n_openvino_model/`).
  - *Why*: Allows immediate, zero-latency interactive testing with your webcam and custom test videos right in the web preview while giving you the exact Python FastAPI + OpenCV + OpenVINO files required for local deployment on your Intel Core i7 machine.
  - *Alternatives Considered*: Static video mocks (rejected because they cannot process your live webcam or custom uploaded video files) or cloud-only frame inference (rejected because it violates the zero-cloud-latency and CNDP privacy-first edge architecture).

- **Decision 2 — Selective CNDP Law 09-08 Privacy Unmasking State Machine**:
  - *Chosen Approach*: Maintain a per-subject state machine where `privacy_masked = True` is enforced by default on the top 28% head/face ROI of every person bounding box (`cv2.GaussianBlur(roi, (51, 51), 30)` in Python and multi-pass Gaussian canvas filter in the web engine). Only when `alert_active == True` for that specific track ID does `privacy_masked` flip to `False`, logging an immutable timestamped entry in the CNDP Compliance Audit Trail.
  - *Why*: Demonstrates verifiable legal compliance with Moroccan Law 09-08 (protection of individuals with regard to the processing of personal data) while ensuring emergency responders can visually assess a collapsed pedestrian or armed intruder.

---

### 4. Technical Architecture & Data Strategy *(Technical Reference)*

- **Architecture & Component Diagram**:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         SAHRA-EYE SMART CITY COMMAND CONSOLE                         │
│   Top Bar: [Sahra-Eye] ── [Live Command | Corridor Map | Webhooks | Python Code]     │
└───────────────┬──────────────────────────────────────────────────────┬───────────────┘
                │                                                      │
                ▼                                                      ▼
┌───────────────────────────────────────────┐        ┌─────────────────────────────────┐
│     VIDEO INGESTION & EDGE CV CANVAS      │        │  LAÂYOUNE TACTICAL TELEMETRY    │
│ ┌───────────────────────────────────────┐ │        │ ┌─────────────────────────────┐ │
│ │ Input Source Controller               │ │        │ │ Boulevard Mekka Interactive │ │
│ │ • Live Webcam (getUserMedia)          │ │        │ │ Vector Map & Camera Nodes   │ │
│ │ • Custom Video Upload (MP4/WebM)      │ │        │ │ • Green-Wave Signal Sync    │ │
│ │ • Interactive Incident Simulator      │ │        │ │ • Protection Civile Route   │ │
│ └──────────────────┬────────────────────┘ │        │ └──────────────┬──────────────┘ │
│                    ▼                      │        │                ▼                │
│ ┌───────────────────────────────────────┐ │        │ ┌─────────────────────────────┐ │
│ │ Real-Time Vision Pipeline (<canvas>)  │ │        │ │ Frugal Webhook Dispatcher   │ │
│ │ • CNDP-0908 Gaussian Face Blur ROI    │ │        │ │ • WhatsApp / Telegram JSON  │ │
│ │ • Mode 1: Pose Aspect/Velocity Fall   │ │───────►│ │ • CNDP-0908 Unmask Audit    │ │
│ │ • Mode 2: Torso HSV Color Threshold   │ │ Alerts │ └─────────────────────────────┘ │
│ │ • Mode 3: Mask Concealment & Weapon   │ │        └─────────────────────────────────┘
│ └───────────────────────────────────────┘ │
└───────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                 DOWNLOADABLE PYTHON EDGE ENGINE (OpenVINO + FastAPI)                 │
│  ┌──────────────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │ vision_engine.py (SahraEyeEngine)    │    │ app.py (FastAPI Server)             │ │
│  │ • YOLOv8n + YOLOv8n-Pose -> OpenVINO │◄───│ • GET /video_feed (MJPEG 30+ FPS)   │ │
│  │ • Intel Core i7 CPU/iGPU AUTO device │    │ • GET /telemetry (JSON State)       │ │
│  │ • HSV Torso Mask & Fall Pose Math    │    │ • POST /mode & /target_color        │ │
│  └──────────────────────────────────────┘    └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

- **Data Model & State**:
  - **EngineState (`TelemetryPayload`)**:
    - `activeMode`: `'najda' | 'tifawt' | 'security'`
    - `targetColor`: `'RED' | 'BLUE' | 'YELLOW' | 'BLACK' | 'WHITE' | 'GREEN'`
    - `inputSource`: `'webcam' | 'upload' | 'simulation'`
    - `fps`: `number` (tabular real-time frame rate counter)
    - `inferenceMs`: `number` (OpenVINO / Canvas frame latency in milliseconds)
    - `cndpMaskedCount`: `number` (citizens currently protected by Gaussian blur)
    - `unmaskedAlertCount`: `number` (bounding boxes unmasked due to critical trigger)
    - `greenWaveActive`: `boolean` (whether Boulevard Mekka traffic corridor is synchronized)
    - `greenWaveNodes`: Array of 5 Boulevard Mekka intersections (`Place El Méchouar`, `Croisement Av. Hassan II`, `Rond-Point Bd Mekka Central`, `Carrefour Hôpital Moulay Hassan`, `Accès Protection Civile`) with live signal state (`RED | GREEN_WAVE`) and ambulance ETA.
  - **DetectedSubject**:
    - `id`: Track identifier
    - `bbox`: `[x, y, width, height]`
    - `headRoi`: `[hx, hy, hw, hh]`
    - `keypoints`: 17 COCO pose coordinates (nose, shoulders, hips, knees, ankles)
    - `dominantTorsoColor`: Detected HSV color classification + match percentage
    - `isFallen`: Boolean computed from bounding-box width-to-height ratio (`w / h > 1.25`) and shoulder-to-hip vertical collapse
    - `hasSecurityThreat`: Boolean computed from upper-face concealment or weapon detection
    - `cndpBlurred`: Boolean (`!isFallen && !hasSecurityThreat`)
  - **WebhookDispatchEvent**:
    - `id`, `timestamp`, `channel` (`'WhatsApp Cloud API' | 'Telegram Bot API'`), `mode`, `cameraNode`, `gpsCoordinates`, `cndpUnmaskReason`, and full `jsonPayload`.

- **Interactive Component & State Mapping**:
  - **Video Source Switcher**: Clicking `Use My Webcam` requests camera permissions via `navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })` and pipes frames into the vision processor; clicking `Upload Test Video` opens a file input accepting `.mp4, .webm, .mov` and runs the real-time pixel HSV and bounding-box analyzer on the uploaded video playback; built-in playback controls allow pausing, scrubbing, and injecting test subjects (e.g., triggering a pedestrian collapse, spawning subjects with Red/Blue/Yellow/Green/Black/White shirts, or simulating a masked intruder in a restricted zone).
  - **Mode & HSV Color Selector**: Switching modes immediately updates the active CV pipeline rules. In **Mode 2 (Tifawt-Guard)**, clicking any of the 6 target colors (`Red`, `Blue`, `Yellow`, `Black`, `White`, `Green`) adjusts the live HSV threshold filter and highlights matching torso regions in the video stream while keeping non-critical faces blurred under CNDP-0908.
  - **Boulevard Mekka Interactive Map**: Clicking any camera or traffic intersection node on the Laâyoune map switches the active camera telemetry context or manually tests the Green-Wave signal cascade along Boulevard Mekka.
  - **Webhook & CNDP Audit Inspector**: Selecting any dispatched alert inspects the exact JSON webhook payload ready for WhatsApp/Telegram integration, with one-click copy and manual webhook test dispatch.
  - **Python Source Code Exporter**: Provides full syntax-highlighted views and one-click `.py` file downloads for `vision_engine.py`, `app.py`, and `requirements.txt`.

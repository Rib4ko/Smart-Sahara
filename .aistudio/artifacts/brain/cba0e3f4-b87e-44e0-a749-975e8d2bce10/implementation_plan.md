# Sahra-Eye: Dual-Scale Adaptive Vision & Indefinite Faint-Lock Architecture

This revised plan upgrades the Sahra-Eye computer vision pipeline and incorporates all three technical risk mitigations: **Trailing Peak-Height Hysteresis** for scene-scale locking, **1.5–2.0s Temporal Immobility Verification** to distinguish true faints from bending/sitting, and **IoU-Gated Soft Optical-Flow Blending** to eliminate asynchronous bounding-box snapping.

## User Review & Critical Decisions

> [!IMPORTANT]
> This revision integrates the three blind-spot mitigations directly into the core state machine so neither mid-collapse height shrink nor network latency can cause mode flipping, false crouch triggers, or box jumping.

- **Confirmed Decision 1 — Scene-Scale Router with 60-Frame Peak-Height Hysteresis**: Instead of evaluating the $28\%$ height threshold on a single instantaneous frame (which could drop below $28\%$ when a close-up person collapses forward onto the floor), the router tracks the **rolling peak subject height over a trailing 60-frame window (plus session peak memory)**. Once `Close-Up Mode` is triggered, hysteresis locks the scene in `Close-Up Mode` throughout the collapse so it can never flip to `Wide Crowd Mode` mid-fall.
- **Confirmed Decision 2 — 1.5–2.0s Temporal Immobility Gate Before Immutable Faint Lock**: To prevent false alarms when someone casually bends down to tie a shoe or pick up an object, mechanical collapse signals (head drop $\ge 18\%$, height compression $\ge 28\%$, floor-zone entry) first enter a 1.5–2.0 second **Pending Floor Verification (`Analyzing Potential Collapse...`)** stage requiring continuous floor-zone presence and low kinetic motion (immobility) before committing to the immutable Red `FAINT DETECTED` lock.
- **Confirmed Decision 3 — IoU-Gated ($\text{IoU} > 0.35$) Soft Blending for Asynchronous Neural Updates**: Server-side `/api/analyze-frame` responses carry their capture timestamp and are reconciled using **IoU overlap gating ($\text{IoU} > 0.35$)** and smooth Exponential Moving Average (EMA) position blending rather than hard coordinate overwriting, preventing bounding boxes from snapping or teleporting when students run across the gym during a 400–1000 ms inference round-trip.

---

## 1. Overview & Core Concept

- **What It Does**: Provides an intelligent, real-time Smart City and School safety monitoring console for Laâyoune, Morocco that processes uploaded test videos, live webcam feeds, and simulated scenarios to detect faints/medical collapses, locate missing persons by clothing color, and guard restricted zones.
- **Target Audience / Persona**: Municipal emergency dispatchers (Protection Civile Laâyoune), school safety coordinators (Lycée Hassan II), and computer vision engineers testing edge-to-cloud video analytics.
- **Key Value**: Delivers reliable, zero-jitter person and faint tracking across any camera angle while automatically clearing a 5-turn **Hospital $\rightarrow$ School Green Zone** and stopping opposing traffic in **Red** with live ambulance tracking.

---

## 2. User Experience & Visual Design

- **Key User Flows**:
  1. **Upload & Auto-Scale Lock**: The operator uploads a close-up room video or a wide-angle school gymnasium video. The engine evaluates the rolling 60-frame peak subject height and locks the appropriate scale mode (`Close-Up Room` vs. `Wide-Angle Crowd`).
  2. **Standing & Active Tracking**: Standing individuals are tracked with smooth bounding boxes and proportionally scaled, EMA-stabilized skeletons.
  3. **Two-Stage Faint Verification (1.5–2.0s Immobility Gate $\rightarrow$ Immutable Lock)**:
     - *Stage A (Collapse / Bending Detected, $0.0\text{s} \rightarrow 1.6\text{s}$)*: Bounding box shifts to Amber (`Verifying Floor Immobility...`) while measuring whether the person stays on the floor with near-zero velocity.
     - *Stage B (Confirmed Faint, $\ge 1.6\text{s}$ Continuous Floor Immobility)*: Commits to the immutable Crimson Red (`#EF4444`) `FAINT DETECTED` lock, freezes the zero-jitter anatomical floor skeleton inside the box indefinitely, and displays the **Last Standing Position** ghost box.
  4. **Automated Laâyoune Green-Wave Dispatch**: Upon confirmed faint lock, the 5-turn route from **Hôpital Moulay Hassan** to **Lycée Hassan II School** turns into a glowing **Green Zone**, opposing and cross streets turn **Red** (stopping opposing vehicles), and the **Ambulance GPS Tracker (`AMB-LAA-01`)** navigates all 5 turns to the school.
- **Visual Identity & Theme**:
  - *Aesthetic Direction*: Clean, high-contrast municipal operations dashboard with plain-English controls and zero technical clutter.
  - *Color Palette & Mood*: Deep slate canvas (`#0B0F17`), structural card surfaces (`#111827`) with `1px` borders (`#1E293B`), cyan primary controls (`#06B6D4`), emerald green (`#10B981`) for safe standing states and open ambulance corridors, amber (`#F59E0B`) for pending fall verification and clothing matches, and crimson (`#EF4444`) for locked faint alerts and stopped cross-traffic.
  - *Typography & Hierarchy*: `Plus Jakarta Sans` for crisp UI headings and labels, paired with `JetBrains Mono` (`tabular-nums`) for FPS counters, video timestamps, and ambulance ETA countdowns.
  - *Component Styling & Layout*: 1440px desktop container with a 3-zone top navigation bar, 3 plain-language mode selector cards, and a balanced 12-column workspace (7 columns for the Vision Player, 5 columns for the Live Alert Banner, 5-Turn Laâyoune Traffic Map, and Mobile Alert Log).
- **Interactive Feedback & Motion**: Smooth timeline scrubbing for uploaded videos, visual countdown progress during the 1.5–2.0s immobility verification window, pulsing locator ring on fainted individuals in crowd scenes, and 30 FPS traffic simulation on the Laâyoune map.

---

## 3. Key Product Decisions & Blind-Spot Mitigations

| Component | Potential Risk | Implemented Mitigation in Architecture |
| :--- | :--- | :--- |
| **Scale Threshold Transition ($\ge 28\%$)** | If a person collapses forward or toward the camera, their instantaneous vertical height can drop below $28\%$, inadvertently triggering a mid-video switch to `Wide Crowd Mode`. | **60-Frame Trailing Peak-Height Hysteresis**: Scale mode is governed by $\max_{t-60 \le k \le t}(h_k)$ and latched once `Close-Up Mode` is entered. Even if a collapsing subject compresses to $14\%$ of frame height on the floor, the scene remains locked in `Close-Up Mode`. |
| **Heuristic Triggers for Close-Up Faints** | Mechanical rules (head drop $\ge 18\%$, height compression $\ge 28\%$) could misfire if a person casually sits on the floor or bends down to pick something up. | **1.5–2.0s Temporal Immobility Gate**: Requires continuous floor-zone posture **plus** low kinetic pixel energy (immobility) for $1.5\text{–}2.0\text{ seconds}$ before committing to the immutable Red faint lock. Brief bends or crouches that recover within $<1.5\text{s}$ auto-cancel cleanly. |
| **Asynchronous API Call Latency** | In wide-angle mode, server-side inference takes $400\text{–}1000\text{ ms}$. If KLT optical flow moves a student while waiting for the response, overwriting with stale coordinates causes boxes to snap/jitter. | **IoU-Gated ($\text{IoU} > 0.35$) Soft Interpolation**: KLT optical flow drives 30 FPS visual motion. Returning `/api/analyze-frame` boxes only update an existing moving track if $\text{IoU} > 0.35$ (blended smoothly at $\alpha = 0.25$), while forward-projecting the box by the track's optical-flow displacement $\Delta x, \Delta y$ accumulated since the snapshot timestamp. |

---

## 4. Technical Architecture & Data Strategy *(Technical Reference)*

### Architecture & Component Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT & FRAME CAPTURE                         │
│        Uploaded Video File (Primary)  │  Live Webcam  │  Demo Scene        │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │ 960×540 Frame @ 30 FPS
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│          DUAL-SCALE SCENE ROUTER (WITH 60-FRAME PEAK HYSTERESIS)           │
│   Rolling Peak Height Buffer: H_peak = max(height[t - 60 ... t], H_latched)│
│   ├── H_peak >= 28% Frame H ──► CLOSE-UP MODE (Latched Hysteresis)         │
│   │                             • Sub-tile slicing & small blobs DISABLED  │
│   │                             • Containment NMS merges nested sub-boxes  │
│   └── H_peak <  28% Frame H ──► WIDE CROWD MODE (25-Student Gym Pipeline)  │
│                                 • KLT Optical Flow @ 30 FPS                │
│                                 • Async /api/analyze-frame (IoU > 0.35)    │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│        TWO-STAGE TEMPORAL IMMOBILITY & INDEFINITE FAINT-LOCK ENGINE        │
│  1. Collapse Candidate Detected (Head Drop / Compression / Prone Aspect)   │
│  2. Temporal Immobility Gate (1.5–2.0s continuous floor presence + low v)  │
│     ├── Moves / Stands Up < 1.5s ──► Cancel candidate, resume Standing     │
│     └── Stationary on Floor >= 1.5s ──► COMMIT IMMUTABLE FAINT LOCK        │
│         • Locks faintLockedBbox & zero-jitter floor skeleton indefinitely  │
│         • Unlocks ONLY if person stands back up (15+ upright frames)       │
└───────────────────┬───────────────────────────────────┬────────────────────┘
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────────┐
│     VISION CANVAS OVERLAY RENDERER    │ │   LAÂYOUNE 5-TURN TRAFFIC MAP    │
│ • Proportional EMA Skeletons          │ │ • Hospital -> School GREEN ZONE  │
│ • Amber Verification -> Red Lock      │ │ • Opposite & Cross Routes RED    │
│ • Pulsing Red Faint Beacon            │ │ • Live Ambulance GPS Tracker     │
└───────────────────────────────────────┘ └──────────────────────────────────┘
```

### Data Model & State

- **`ScaleHysteresisState`**:
  - `trailingMaxHeights: number[]` (ring buffer of last 60 frames)
  - `sessionPeakHeightRatio: number` (latches `Close-Up Mode` once $\ge 0.28$, preventing mid-collapse mode switches)
- **`TrackedPersonState`**:
  - `id`, `label`, `bbox`, `standingBaselineBbox`, `peakHeight`, `baselineTopY`
  - `collapseCandidateStartMs: number | null` — tracks elapsed time in the floor/compressed posture
  - `floorImmobilityScore: number` — measures low kinetic motion during the $1.5\text{–}2.0\text{s}$ verification window
  - `isVerifyingFall: boolean` — true during the $0.3\text{s}\text{–}1.6\text{s}$ Amber verification window
  - `isFallen: boolean`, `faintLockedBbox: [number, number, number, number] | null` — committed immutable Red lock once temporal persistence ($\ge 1.6\text{s}$) is satisfied
- **`AsyncSnapshotHeader`**:
  - Records `snapshotTimestampMs` and per-track positions at capture time so when `/api/analyze-frame` returns $400\text{–}1000\text{ ms}$ later, each returned box is compensated by the track's accumulated optical-flow drift $(\Delta x, \Delta y)$ and gated by $\text{IoU} > 0.35$.

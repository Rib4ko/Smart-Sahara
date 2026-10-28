# Implementation Plan: Shadow-Cropped Foreground Tracking & Sub-1s Faint Lock for Place du Méchouar Video

## 1. Root Cause Analysis of the Test Video (`00:00–00:09`)

Frame-by-frame inspection of the uploaded **Place du Méchouar (`CAM 04 / 05`)** video reveals why the bounding box was jumping around:

1. **Wide-Plaza Foreground Perspective vs. Indoor Floor Rule (`00:00–00:03`)**
   - While the man in the cream djellaba is standing and walking in the foreground (`x ≈ 52%–60%, y ≈ 60%–87%`), his head is already at `y ≈ 60%` of the frame (`bh ≈ 26%` of frame height, vertical aspect ratio `bw / bh ≈ 0.42`).
   - The previous indoor rule (`by > 46% of frame height = floor zone`) falsely classified him as already in the floor zone while he was still standing upright.

2. **Cast Ground Shadow & Background Motion Distractors (`00:00–00:09`)**
   - In bright sunlight, the subject casts a long dark shadow to the left (`x ≈ 41%–54%, y ≈ 83%–87%`).
   - Simultaneously, waving Moroccan flags in the upper-right (`x ≈ 75%–95%, y ≈ 20%–32%`), distant tiny pedestrians (`x ≈ 14%, y ≈ 42%`), and blinking CCTV overlays (`2026-10-28 15:41` top-right, `CAM 04 / 05` bottom-right) create continuous frame-wide pixel differences.
   - When the subject collapses horizontally in a loose djellaba (`00:04–00:09`) and MediaPipe briefly drops pose confidence, the fallback global motion box (`motionMinX..motionMaxX`) and dark-pixel check (`luma < 62`) latched onto his cast shadow and the waving background flags, causing the detection box to stretch and jump.

3. **Unclamped Spatial Matching**
   - Single-subject mode matched any candidate box without enforcing a spatial distance gate around the person's last known position, allowing momentary background detections (monument archway, flags, or distant walkers) to pull the bounding box away from the subject.

---

## 2. Proposed Architecture Upgrades

### A. Primary Foreground Subject Selection & CCTV Masking
- **Foreground Dominance Filter**: When a primary foreground person (`height >= 12%` of frame or `area >= 1.2%` of frame) is tracked, automatically ignore tiny background walkers (`height < 9%` of frame) and static monument structures.
- **CCTV Overlay & Sky/Flag Exclusion Zones**: Exclude top-right timestamp overlays (`y < 10%`), bottom-right camera ID overlays (`y > 91%, x > 80%`), and upper-horizon waving flags (`y < 36%`) from local motion/foreground refinement.
- **Upgraded Pose Landmarker**: Use MediaPipe `pose_landmarker_full` (with automatic fallback to `pose_landmarker_lite`) for stronger keypoint tracking on loose robes/djellabas and horizontal body postures.

### B. Shadow-Cropped Local ROI Body Tracker (Zero Box Jumping)
- **Spatially Gated Local Search Window**: Once the primary person is detected, anchor a local tracking window (`cx ± 16% W, cy ± 16% H`) around their last verified position so the bounding box can **never** jump to distant background objects.
- **Automatic Ground-Shadow Cropping**:
  - Sample the surrounding plaza ground luminance and chromaticity `(bgR, bgG, bgB, bgLuma)` around the person's bounding box.
  - Identify and strip out ground shadow pixels (`luma < bgLuma * 0.76` with low chromatic saturation `S < 65` extending horizontally along the bottom ground plane) while preserving the actual human body silhouette (cream/beige djellaba, yellow headwrap, skin, or colored clothing).
  - Apply tight morphological percentile bounds (`5th–95th` X/Y percentiles) with velocity-clamped EMA smoothing so the bounding box hugs the walking and fallen person smoothly on every single frame.

### C. Relative Posture Collapse Geometry & `< 1.0s` Indefinite Faint Lock
- **Relative Standing Baseline (Camera-Angle Independent)**:
  - While the person is upright (`aspect = bw / bh < 0.78`), record their rolling standing baseline (`standingTopY`, `standingBottomY`, `peakHeight`).
  - Detect a fall/faint relative to the person's own standing baseline (works whether the person is at `y = 20%` indoors or `y = 60%–87%` on an outdoor plaza):
    1. **Horizontal Aspect Flip**: `bw / bh` increases from `< 0.78` to `>= 1.15` (reaches `~2.4` at `00:05` in the test video),
    2. **Relative Height Compression**: `bh <= peakHeight * 0.62` while the bottom edge remains anchored near `standingBottomY`,
    3. **Relative Head/Torso Drop**: top `by` drops by `>= 35%` of `peakHeight` toward the ground plane (`y` shifts from `60%` down to `77%`).
- **Sub-1-Second Faint Lock (`0.75s`)**:
  - As soon as the person collapses onto the ground (`00:04–00:05`) and remains down for **`0.75s`** (`< 1 second`), lock the crimson **`FAINT DETECTED`** bounding box and horizontal floor skeleton (`buildFrozenFloorKeypoints`) indefinitely at their exact body position (with zero jitter for the rest of the video) and immediately activate the **Hôpital Moulay Hassan → Place du Méchouar Green-Wave Corridor**.

---

## 3. Files to Modify

- **`/src/components/VisionCanvas.tsx`**:
  - Upgrade MediaPipe initialization to `pose_landmarker_full` (with `lite` fallback).
  - Remove the global motion bounding box and dark-shadow trap (`luma < 62`).
  - Implement `extractShadowCroppedBodyBox()` for local ROI foreground segmentation with outdoor cast-shadow cropping and CCTV/flag zone masking.
  - Enforce spatial distance gating on track matching so background noise can never pull the primary track.
  - Replace hardcoded indoor `by > 0.46 * H` floor checks with relative baseline collapse geometry (`aspect >= 1.15`, `heightCompression >= 38%`, `headDrop >= 35% of peakHeight`) and set the faint confirmation lock threshold to **`0.75s`** (`< 1.0s`).
- **`/server.ts`**:
  - Refine the `/api/analyze-frame` prompt to explicitly instruct the model to return a tight bounding box around the person's physical body only, strictly excluding cast ground shadows on outdoor plazas.

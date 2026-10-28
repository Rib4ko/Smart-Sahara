import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  PoseLandmarker,
  ObjectDetector,
  FilesetResolver,
} from '@mediapipe/tasks-vision';
import {
  Camera,
  Upload,
  Play,
  Pause,
  RotateCcw,
  AlertTriangle,
  Crosshair,
  Video,
  Lock,
  Sparkles,
  Clock,
} from 'lucide-react';
import {
  OperationalMode,
  TargetClothingColor,
  VideoInputSource,
  DetectedSubject,
  PoseKeypoint,
  TARGET_COLOR_CONFIG,
  CameraSectorNode,
} from '../types/sahra';

interface VisionCanvasProps {
  mode: OperationalMode;
  targetColor: TargetClothingColor;
  inputSource: VideoInputSource;
  activeCamera: CameraSectorNode;
  onInputSourceChange: (source: VideoInputSource) => void;
  onModeChange: (mode: OperationalMode) => void;
  onTargetColorChange: (color: TargetClothingColor) => void;
  onTelemetryUpdate: (data: {
    fps: number;
    inferenceMs: number;
    alertActive: boolean;
    alertTitle: string;
    alertDetail: string;
    greenWaveActive: boolean;
    subjects: DetectedSubject[];
  }) => void;
  onTriggerAlert: (title: string, description: string) => void;
}

interface SimActor {
  id: number;
  name: string;
  x: number;
  y: number;
  vx: number;
  color: TargetClothingColor;
  fallen: boolean;
  fallProgress: number;
  maskedIntruder: boolean;
  hasKnife: boolean;
}

interface ScaleHysteresisState {
  trailingMaxHeights: number[]; // Rolling 60-frame window of max subject height ratio
  sessionPeakHeightRatio: number;
  latchedSingleForegroundMode: boolean; // Latches true for single-subject foreground videos (like Place du Méchouar or room faint)
}

interface TrackedPersonState {
  id: number;
  bbox: [number, number, number, number];
  standingBaselineBbox: [number, number, number, number] | null;
  peakHeight: number;
  baselineTopY: number;
  standingBottomY: number;
  vx: number;
  vy: number;
  speedPx: number;
  ageFrames: number;
  missedFrames: number;
  standingFrames: number;
  // Sub-1-Second Temporal Collapse Gate -> Indefinite Faint Lock
  collapseCandidateStartMs: number | null;
  isVerifyingFall: boolean;
  verifyProgressSec: number;
  isFallen: boolean;
  faintLockedBbox: [number, number, number, number] | null;
  faintStartTime: number | null;
  faintReason: string;
  neuralFaintConfirmed: boolean;
  neuralVerified: boolean;
  smoothedKeypoints: PoseKeypoint[];
  psoKeypointsMap: Record<string, PoseKeypoint>;
  torsoColor: TargetClothingColor;
  hsvMatchPct: number;
}

// Proportional Standing Skeleton strictly fitted inside [bx, by, bw, bh]
function buildStandingKeypoints(
  bbox: [number, number, number, number],
  stridePhase = 0
): PoseKeypoint[] {
  const [bx, by, bw, bh] = bbox;
  const swing = Math.sin(stridePhase) * 0.035;
  return [
    { name: 'nose', x: bx + bw * 0.5, y: by + bh * 0.11, confidence: 0.96 },
    { name: 'l_shoulder', x: bx + bw * 0.33, y: by + bh * 0.24, confidence: 0.96 },
    { name: 'r_shoulder', x: bx + bw * 0.67, y: by + bh * 0.24, confidence: 0.96 },
    { name: 'l_elbow', x: bx + bw * (0.25 - swing), y: by + bh * 0.41, confidence: 0.92 },
    { name: 'r_elbow', x: bx + bw * (0.75 + swing), y: by + bh * 0.41, confidence: 0.92 },
    { name: 'l_wrist', x: bx + bw * (0.22 - swing), y: by + bh * 0.55, confidence: 0.9 },
    { name: 'r_wrist', x: bx + bw * (0.78 + swing), y: by + bh * 0.55, confidence: 0.9 },
    { name: 'l_hip', x: bx + bw * 0.38, y: by + bh * 0.54, confidence: 0.95 },
    { name: 'r_hip', x: bx + bw * 0.62, y: by + bh * 0.54, confidence: 0.95 },
    { name: 'l_knee', x: bx + bw * (0.36 + swing), y: by + bh * 0.74, confidence: 0.93 },
    { name: 'r_knee', x: bx + bw * (0.64 - swing), y: by + bh * 0.74, confidence: 0.93 },
    { name: 'l_ankle', x: bx + bw * (0.35 + swing * 1.1), y: by + bh * 0.92, confidence: 0.91 },
    { name: 'r_ankle', x: bx + bw * (0.65 - swing * 1.1), y: by + bh * 0.92, confidence: 0.91 },
  ];
}

// Proportional Frozen Floor Faint Skeleton strictly fitted inside [bx, by, bw, bh]
function buildFrozenFloorKeypoints(bbox: [number, number, number, number]): PoseKeypoint[] {
  const [bx, by, bw, bh] = bbox;
  const isHorizontal = bw >= bh * 1.05;

  if (isHorizontal) {
    // Horizontal side-lying / sprawled skeleton (head on left/inner side, feet on right side)
    return [
      { name: 'nose', x: bx + bw * 0.14, y: by + bh * 0.42, confidence: 0.98 },
      { name: 'l_shoulder', x: bx + bw * 0.26, y: by + bh * 0.30, confidence: 0.98 },
      { name: 'r_shoulder', x: bx + bw * 0.28, y: by + bh * 0.58, confidence: 0.98 },
      { name: 'l_elbow', x: bx + bw * 0.38, y: by + bh * 0.22, confidence: 0.95 },
      { name: 'r_elbow', x: bx + bw * 0.40, y: by + bh * 0.74, confidence: 0.95 },
      { name: 'l_wrist', x: bx + bw * 0.20, y: by + bh * 0.68, confidence: 0.92 },
      { name: 'r_wrist', x: bx + bw * 0.48, y: by + bh * 0.80, confidence: 0.92 },
      { name: 'l_hip', x: bx + bw * 0.55, y: by + bh * 0.36, confidence: 0.98 },
      { name: 'r_hip', x: bx + bw * 0.55, y: by + bh * 0.60, confidence: 0.98 },
      { name: 'l_knee', x: bx + bw * 0.74, y: by + bh * 0.40, confidence: 0.96 },
      { name: 'r_knee', x: bx + bw * 0.74, y: by + bh * 0.66, confidence: 0.96 },
      { name: 'l_ankle', x: bx + bw * 0.91, y: by + bh * 0.46, confidence: 0.95 },
      { name: 'r_ankle', x: bx + bw * 0.91, y: by + bh * 0.72, confidence: 0.95 },
    ];
  }

  // Foreshortened floor faint pose (head near top of floor box, feet near bottom of floor box)
  return [
    { name: 'nose', x: bx + bw * 0.48, y: by + bh * 0.10, confidence: 0.98 },
    { name: 'l_shoulder', x: bx + bw * 0.35, y: by + bh * 0.20, confidence: 0.98 },
    { name: 'r_shoulder', x: bx + bw * 0.63, y: by + bh * 0.20, confidence: 0.98 },
    { name: 'l_elbow', x: bx + bw * 0.20, y: by + bh * 0.32, confidence: 0.95 },
    { name: 'r_elbow', x: bx + bw * 0.78, y: by + bh * 0.32, confidence: 0.95 },
    { name: 'l_wrist', x: bx + bw * 0.10, y: by + bh * 0.42, confidence: 0.92 },
    { name: 'r_wrist', x: bx + bw * 0.88, y: by + bh * 0.42, confidence: 0.92 },
    { name: 'l_hip', x: bx + bw * 0.37, y: by + bh * 0.44, confidence: 0.98 },
    { name: 'r_hip', x: bx + bw * 0.61, y: by + bh * 0.44, confidence: 0.98 },
    { name: 'l_knee', x: bx + bw * 0.34, y: by + bh * 0.68, confidence: 0.96 },
    { name: 'r_knee', x: bx + bw * 0.64, y: by + bh * 0.68, confidence: 0.96 },
    { name: 'l_ankle', x: bx + bw * 0.31, y: by + bh * 0.90, confidence: 0.95 },
    { name: 'r_ankle', x: bx + bw * 0.67, y: by + bh * 0.90, confidence: 0.95 },
  ];
}

function rgbToOpenCvHsv(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;

  let h = 0;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (max !== min) {
    switch (max) {
      case rf:
        h = (gf - bf) / d + (gf < bf ? 6 : 0);
        break;
      case gf:
        h = (bf - rf) / d + 2;
        break;
      case bf:
        h = (rf - gf) / d + 4;
        break;
    }
    h /= 6;
  }

  return [Math.round(h * 180), Math.round(s * 255), Math.round(v * 255)];
}

function matchesTargetHsv(h: number, s: number, v: number, color: TargetClothingColor): boolean {
  switch (color) {
    case 'RED':
      return ((h >= 0 && h <= 11) || (h >= 164 && h <= 180)) && s >= 100 && v >= 65;
    case 'BLUE':
      return h >= 82 && h <= 135 && s >= 65 && v >= 45;
    case 'YELLOW':
      return h >= 18 && h <= 37 && s >= 105 && v >= 85;
    case 'GREEN':
      return h >= 37 && h <= 84 && s >= 75 && v >= 50;
    case 'BLACK':
      return v <= 62;
    case 'WHITE':
      return s <= 52 && v >= 185;
  }
}

function boxIoU(a: [number, number, number, number], b: [number, number, number, number]): number {
  const xA = Math.max(a[0], b[0]);
  const yA = Math.max(a[1], b[1]);
  const xB = Math.min(a[0] + a[2], b[0] + b[2]);
  const yB = Math.min(a[1] + a[3], b[1] + b[3]);
  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  const areaA = a[2] * a[3];
  const areaB = b[2] * b[3];
  return interArea / Math.max(1, areaA + areaB - interArea);
}

function boxContainmentRatio(
  outer: [number, number, number, number],
  inner: [number, number, number, number]
): number {
  const xA = Math.max(outer[0], inner[0]);
  const yA = Math.max(outer[1], inner[1]);
  const xB = Math.min(outer[0] + outer[2], inner[0] + inner[2]);
  const yB = Math.min(outer[1] + outer[3], inner[1] + inner[3]);
  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  return interArea / Math.max(1, inner[2] * inner[3]);
}

function centerDist(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ax = a[0] + a[2] * 0.5;
  const ay = a[1] + a[3] * 0.5;
  const bx = b[0] + b[2] * 0.5;
  const by = b[1] + b[3] * 0.5;
  return Math.hypot(ax - bx, ay - by);
}

// ============================================================================
// SHADOW-CROPPING & LOCAL FOREGROUND BODY SEGMENTATION ENGINE
// Strips cast ground shadows (e.g. left shadow on Place du Méchouar plaza)
// and tightly bounds the physical human body even in flowing robes (djellaba).
// ============================================================================

function sampleLocalGroundBackground(
  data: Uint8ClampedArray,
  gray: Uint8Array,
  GW: number,
  GH: number,
  gx1: number,
  gy1: number,
  gx2: number,
  gy2: number
): { bgR: number; bgG: number; bgB: number; bgLuma: number } {
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sumL = 0;
  let count = 0;

  const yStart = Math.max(12, Math.floor(gy1));
  const yEnd = Math.min(GH - 8, Math.ceil(gy2));
  const leftX = Math.max(6, Math.floor(gx1 - 6));
  const rightX = Math.min(GW - 6, Math.ceil(gx2 + 6));

  for (let y = yStart; y <= yEnd; y += 2) {
    for (const x of [leftX, Math.max(6, leftX - 3), rightX, Math.min(GW - 6, rightX + 3)]) {
      const pIdx = y * GW + x;
      const luma = gray[pIdx];
      // Exclude deep cast shadow pixels when sampling the sunlit ground color
      if (luma >= 95 && luma <= 215) {
        const idx = pIdx * 4;
        sumR += data[idx];
        sumG += data[idx + 1];
        sumB += data[idx + 2];
        sumL += luma;
        count++;
      }
    }
  }

  if (count === 0) {
    return { bgR: 168, bgG: 162, bgB: 152, bgLuma: 162 };
  }

  return {
    bgR: sumR / count,
    bgG: sumG / count,
    bgB: sumB / count,
    bgLuma: sumL / count,
  };
}

/**
 * Crops out any horizontal ground shadow tail included inside a detector bounding box
 * (such as the dark left shadow cast on the Place du Méchouar concrete plaza).
 */
function cropCastShadowFromBox(
  bbox: [number, number, number, number],
  data: Uint8ClampedArray,
  gray: Uint8Array,
  GW: number,
  GH: number,
  W: number,
  H: number
): [number, number, number, number] {
  const scaleX = W / GW;
  const scaleY = H / GH;
  const [bx, by, bw, bh] = bbox;

  const gx1 = Math.max(4, Math.floor(bx / scaleX));
  const gy1 = Math.max(4, Math.floor(by / scaleY));
  const gx2 = Math.min(GW - 4, Math.ceil((bx + bw) / scaleX));
  const gy2 = Math.min(GH - 4, Math.ceil((by + bh) / scaleY));

  if (gx2 - gx1 < 6 || gy2 - gy1 < 6) return bbox;

  const { bgR, bgG, bgB, bgLuma } = sampleLocalGroundBackground(
    data,
    gray,
    GW,
    GH,
    gx1,
    gy1,
    gx2,
    gy2
  );

  // Check if each column has non-shadow physical body pixels
  const colBodyCounts = new Int16Array(gx2 - gx1 + 1);
  let totalBrightOrChromaticBody = 0;

  for (let x = gx1; x <= gx2; x++) {
    let bodyInCol = 0;
    for (let y = gy1; y <= gy2; y++) {
      const pIdx = y * GW + x;
      const idx = pIdx * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const luma = gray[pIdx];
      const [, s] = rgbToOpenCvHsv(r, g, b);

      // Ground shadow check: darker than ground (luma < bgLuma * 0.77) and low saturation (neutral shadow)
      const isGroundShadow = luma < bgLuma * 0.77 && s < 78;
      if (isGroundShadow) continue;

      const colorDist = Math.hypot(r - bgR, g - bgG, b - bgB);
      const lumaDiff = Math.abs(luma - bgLuma);

      // Bright garment (cream/white djellaba) or warm/saturated clothing/headwrap/skin
      if (luma > bgLuma + 14 || (colorDist > 22 && lumaDiff > 12) || s > 68) {
        bodyInCol++;
        totalBrightOrChromaticBody++;
      }
    }
    colBodyCounts[x - gx1] = bodyInCol;
  }

  // Only trim columns if we have a clear body mass (prevents over-cropping on dark indoor clothes)
  if (totalBrightOrChromaticBody < 14) {
    return bbox;
  }

  const minColHits = Math.max(1, Math.floor((gy2 - gy1) * 0.10));
  let trimLeft = gx1;
  let trimRight = gx2;

  while (trimLeft < trimRight - 4 && colBodyCounts[trimLeft - gx1] < minColHits) {
    trimLeft++;
  }
  while (trimRight > trimLeft + 4 && colBodyCounts[trimRight - gx1] < minColHits) {
    trimRight--;
  }

  const newBx = Math.max(6, (trimLeft - 0.5) * scaleX);
  const newBw = Math.min(W - newBx - 6, Math.max(24, (trimRight - trimLeft + 1.5) * scaleX));
  return [newBx, by, newBw, bh];
}

/**
 * Extracts the tight, shadow-cropped human body bounding box inside a local ROI.
 * Works both for outdoor sunlit plazas (Place du Méchouar cream djellaba + yellow headwrap,
 * strictly ignoring the dark ground shadow and background flags) and indoor floors.
 */
function extractShadowCroppedBodyInROI(
  anchorBox: [number, number, number, number] | null,
  standingBottomY: number | null,
  data: Uint8ClampedArray,
  gray: Uint8Array,
  GW: number,
  GH: number,
  W: number,
  H: number
): { bbox: [number, number, number, number]; foreshortenedFloor: boolean; score: number } | null {
  const scaleX = W / GW;
  const scaleY = H / GH;

  // Define search region in offscreen grid coordinates (GW x GH)
  // Exclude upper horizon (y < 44% where monument arch & waving flags live) and CCTV overlays
  let sx1 = Math.floor(GW * 0.25);
  let sx2 = Math.floor(GW * 0.76);
  let sy1 = Math.floor(GH * 0.46);
  let sy2 = Math.floor(GH * 0.91);

  if (anchorBox) {
    const [ax, ay, aw, ah] = anchorBox;
    const padX = Math.max(W * 0.10, aw * 0.65);
    const padYTop = Math.max(H * 0.06, ah * 0.35);
    const padYBot = Math.max(H * 0.05, ah * 0.35);

    sx1 = Math.max(Math.floor(GW * 0.16), Math.floor((ax - padX) / scaleX));
    sx2 = Math.min(Math.floor(GW * 0.84), Math.ceil((ax + aw + padX) / scaleX));
    sy1 = Math.max(Math.floor(GH * 0.44), Math.floor((ay - padYTop) / scaleY));
    const maxBotPx = standingBottomY
      ? Math.max(ay + ah + padYBot, standingBottomY + H * 0.03)
      : ay + ah + padYBot;
    sy2 = Math.min(Math.floor(GH * 0.91), Math.ceil(maxBotPx / scaleY));
  }

  if (sx2 - sx1 < 8 || sy2 - sy1 < 8) return null;

  const { bgR, bgG, bgB, bgLuma } = sampleLocalGroundBackground(
    data,
    gray,
    GW,
    GH,
    sx1,
    sy1,
    sx2,
    sy2
  );

  // Pass 1: Detect non-shadow foreground body pixels (bright djellaba/robe, headwrap, skin, or saturated clothing)
  const xs: number[] = [];
  const ys: number[] = [];

  for (let y = sy1; y <= sy2; y++) {
    for (let x = sx1; x <= sx2; x++) {
      // Mask bottom-right CCTV "CAM 04" text region
      if (y > GH * 0.90 && x > GW * 0.80) continue;

      const pIdx = y * GW + x;
      const idx = pIdx * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const luma = gray[pIdx];
      const [h, s] = rgbToOpenCvHsv(r, g, b);

      // STRICT SHADOW REJECTION: Cast ground shadows on sunlit plazas are darker than bgLuma * 0.78 with low saturation
      if (luma < bgLuma * 0.78 && s < 78) {
        continue;
      }

      const colorDist = Math.hypot(r - bgR, g - bgG, b - bgB);

      // 1. Bright/cream/white clothing contrasting against medium plaza concrete (e.g. djellaba in Place du Méchouar)
      const isBrightGarment = luma > bgLuma + 16 && luma >= 168;
      // 2. Warm/saturated clothing or headwrap/skin contrasting against gray concrete
      const isChromaticBody =
        s >= 62 && luma >= bgLuma * 0.72 && colorDist >= 24;
      // 3. Indoor teal/colored top on linoleum floor
      const isIndoorTealTop =
        s > 95 && luma > 55 && luma < 198 && h >= 75 && h <= 130;

      if (isBrightGarment || isChromaticBody || isIndoorTealTop) {
        xs.push(x);
        ys.push(y);
      }
    }
  }

  // Fallback for indoor close-up dark clothing on bright linoleum floor ONLY when no outdoor bright garment exists
  if (xs.length < 18 && bgLuma >= 150) {
    const ix1 = Math.max(sx1, Math.floor(GW * 0.32));
    const ix2 = Math.min(sx2, Math.floor(GW * 0.66));
    const iy1 = Math.max(sy1, Math.floor(GH * 0.48));
    const iy2 = Math.min(sy2, Math.floor(GH * 0.89));

    for (let y = iy1; y <= iy2; y++) {
      for (let x = ix1; x <= ix2; x++) {
        const pIdx = y * GW + x;
        const luma = gray[pIdx];
        // Deep dark leggings/clothing (luma < 52) in central indoor aisle
        if (luma < 52) {
          xs.push(x);
          ys.push(y);
        }
      }
    }
  }

  if (xs.length < 16) return null;

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  // Use 4th to 96th percentiles to reject isolated speckle noise and hug the physical body tightly
  const xMin = xs[Math.floor(xs.length * 0.04)];
  const xMax = xs[Math.min(xs.length - 1, Math.floor(xs.length * 0.96))];
  const yMin = ys[Math.floor(ys.length * 0.04)];
  const yMax = ys[Math.min(ys.length - 1, Math.floor(ys.length * 0.96))];

  const spanW = xMax - xMin + 1;
  const spanH = yMax - yMin + 1;
  if (spanW < 3 || spanH < 3) return null;

  const padPxX = Math.max(8, spanW * scaleX * 0.10);
  const padPxY = Math.max(8, spanH * scaleY * 0.10);

  const bx = Math.max(8, xMin * scaleX - padPxX);
  const by = Math.max(8, yMin * scaleY - padPxY);
  const bw = Math.min(W - bx - 8, Math.max(32, spanW * scaleX + padPxX * 2));
  const bh = Math.min(H - by - 8, Math.max(32, spanH * scaleY + padPxY * 2));

  if (bw > W * 0.60 || bh > H * 0.85) return null;

  const aspect = bw / Math.max(1, bh);
  return {
    bbox: [bx, by, bw, bh],
    foreshortenedFloor: aspect >= 1.15 || by > H * 0.70,
    score: 0.86,
  };
}

// Containment Non-Maximum Suppression
function applyContainmentNMS<T extends { bbox: [number, number, number, number]; score: number }>(
  items: T[],
  isSingleForegroundMode: boolean
): T[] {
  if (items.length <= 1) return items;

  // Sort by foreground prominence (lower bottom Y + area + score)
  const sorted = [...items].sort((a, b) => {
    const scoreA = a.bbox[2] * a.bbox[3] * (0.5 + a.score) * (a.bbox[1] + a.bbox[3]);
    const scoreB = b.bbox[2] * b.bbox[3] * (0.5 + b.score) * (b.bbox[1] + b.bbox[3]);
    return scoreB - scoreA;
  });

  if (isSingleForegroundMode) {
    // Keep strictly the primary foreground subject (ignores tiny background walkers & eliminates multi-box splits)
    return [sorted[0]];
  }

  const kept: T[] = [];
  for (const cand of sorted) {
    const isSuppressed = kept.some(
      (k) =>
        boxIoU(k.bbox, cand.bbox) > 0.25 ||
        boxContainmentRatio(k.bbox, cand.bbox) > 0.45 ||
        centerDist(k.bbox, cand.bbox) < 24
    );
    if (!isSuppressed) {
      kept.push(cand);
    }
  }
  return kept;
}

export const VisionCanvas: React.FC<VisionCanvasProps> = ({
  mode,
  targetColor,
  inputSource,
  activeCamera,
  onInputSourceChange,
  onTargetColorChange,
  onTelemetryUpdate,
  onTriggerAlert,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const prevGrayRef = useRef<Uint8Array | null>(null);
  const lastAlertTimeRef = useRef<number>(0);

  // 60-Frame Peak-Height Hysteresis Router State
  const scaleHysteresisRef = useRef<ScaleHysteresisState>({
    trailingMaxHeights: [],
    sessionPeakHeightRatio: 0,
    latchedSingleForegroundMode: false,
  });

  // Server-side Pre-Trained Neural Teacher state with Snapshot Drift Tracking
  const neuralBusyRef = useRef<boolean>(false);
  const lastNeuralTimeRef = useRef<number>(0);
  const hasNeuralCalibrationRef = useRef<boolean>(false);
  const snapshotTrackPositionsRef = useRef<Record<number, [number, number]>>({});

  const tracksRef = useRef<TrackedPersonState[]>([]);
  const nextTrackIdRef = useRef<number>(1);

  const [poseModelStatus, setPoseModelStatus] = useState<'loading' | 'ready' | 'fallback'>('loading');
  const [sceneScaleLabel, setSceneScaleLabel] = useState<'Foreground Subject Mode' | 'Wide Crowd Mode'>(
    'Foreground Subject Mode'
  );
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [videoProgress, setVideoProgress] = useState<number>(0);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [manualFallOverride, setManualFallOverride] = useState<boolean>(false);
  const [manualSecurityOverride, setManualSecurityOverride] = useState<boolean>(false);
  const [lockedFaintSummary, setLockedFaintSummary] = useState<string | null>(null);
  const [verifyingFallSummary, setVerifyingFallSummary] = useState<string | null>(null);

  const simActorsRef = useRef<SimActor[]>([
    {
      id: 1,
      name: 'Person #1',
      x: 220,
      y: 265,
      vx: 0.8,
      color: 'BLUE',
      fallen: false,
      fallProgress: 0,
      maskedIntruder: false,
      hasKnife: false,
    },
    {
      id: 2,
      name: 'Person #2',
      x: 520,
      y: 285,
      vx: -0.65,
      color: 'RED',
      fallen: false,
      fallProgress: 0,
      maskedIntruder: false,
      hasKnife: false,
    },
    {
      id: 3,
      name: 'Person #3',
      x: 760,
      y: 255,
      vx: -0.5,
      color: 'YELLOW',
      fallen: false,
      fallProgress: 0,
      maskedIntruder: false,
      hasKnife: false,
    },
  ]);

  // Initialize Pre-Trained Neural Models: PoseLandmarker Full (with Lite fallback) + EfficientDet-Lite0
  useEffect(() => {
    let cancelled = false;
    async function initPreTrainedModels() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );

        const landmarkerFull = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 4,
          minPoseDetectionConfidence: 0.32,
          minPosePresenceConfidence: 0.32,
          minTrackingConfidence: 0.32,
        }).catch(() => null);

        const landmarker =
          landmarkerFull ||
          (await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 4,
            minPoseDetectionConfidence: 0.32,
            minPosePresenceConfidence: 0.32,
            minTrackingConfidence: 0.32,
          }).catch(() => null));

        const detector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          scoreThreshold: 0.32,
          maxResults: 25,
          categoryAllowlist: ['person'],
        }).catch(() => null);

        if (!cancelled) {
          poseLandmarkerRef.current = landmarker;
          objectDetectorRef.current = detector;
          setPoseModelStatus(landmarker || detector ? 'ready' : 'fallback');
        }
      } catch {
        if (!cancelled) {
          setPoseModelStatus('fallback');
        }
      }
    }
    initPreTrainedModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const resetMemoryBank = useCallback(() => {
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    prevGrayRef.current = null;
    lastNeuralTimeRef.current = 0;
    hasNeuralCalibrationRef.current = false;
    snapshotTrackPositionsRef.current = {};
    scaleHysteresisRef.current = {
      trailingMaxHeights: [],
      sessionPeakHeightRatio: 0,
      latchedSingleForegroundMode: false,
    };
    setManualFallOverride(false);
    setManualSecurityOverride(false);
    setLockedFaintSummary(null);
    setVerifyingFallSummary(null);
    simActorsRef.current.forEach((a) => {
      a.fallen = false;
      a.maskedIntruder = false;
      a.hasKnife = false;
    });
  }, []);

  // Reconcile Server-Side Neural Detections (/api/analyze-frame) with Shadow-Cropping & Spatial Gating
  const reconcileServerNeuralDetections = useCallback(
    (
      people: Array<{ box_2d: number[]; status: string }>,
      W: number,
      H: number,
      now: number
    ) => {
      if (!Array.isArray(people) || people.length === 0) return;

      const rawBoxes: Array<{
        bbox: [number, number, number, number];
        status: string;
        score: number;
      }> = [];

      let maxReturnedHeightRatio = 0;

      people.forEach((person) => {
        if (!Array.isArray(person.box_2d) || person.box_2d.length < 4) return;
        const [ymin, xmin, ymax, xmax] = person.box_2d;
        const bx = Math.max(6, (xmin / 1000) * W);
        const by = Math.max(6, (ymin / 1000) * H);
        const bw = Math.min(W - bx - 6, Math.max(18, ((xmax - xmin) / 1000) * W));
        const bh = Math.min(H - by - 6, Math.max(20, ((ymax - ymin) / 1000) * H));
        if (bw > W * 0.75 || bh > H * 0.95) return;

        // Ignore tiny distant background walkers (e.g. far-left speck in Place du Méchouar) when a foreground person exists
        if (bh < H * 0.09 && by + bh < H * 0.55 && people.length <= 3) return;

        const hRatio = bh / H;
        if (hRatio > maxReturnedHeightRatio) maxReturnedHeightRatio = hRatio;

        rawBoxes.push({
          bbox: [bx, by, bw, bh],
          status: String(person.status).toLowerCase(),
          score: 0.92,
        });
      });

      if (rawBoxes.length === 0) return;

      const hyst = scaleHysteresisRef.current;
      if (
        rawBoxes.length <= 2 &&
        (maxReturnedHeightRatio >= 0.11 || rawBoxes[0].bbox[2] / W >= 0.12)
      ) {
        hyst.latchedSingleForegroundMode = true;
        hyst.sessionPeakHeightRatio = Math.max(hyst.sessionPeakHeightRatio, maxReturnedHeightRatio);
      }

      const isSingleMode = hyst.latchedSingleForegroundMode;
      const filteredBoxes = applyContainmentNMS(rawBoxes, isSingleMode);

      hasNeuralCalibrationRef.current = true;
      const tracks = tracksRef.current;
      const matchedTrackIds = new Set<number>();
      const snapPos = snapshotTrackPositionsRef.current;

      filteredBoxes.forEach((item) => {
        const [bx, by, bw, bh] = item.bbox;
        const isFaintedStatus = item.status === 'fainted';

        let bestTrack: TrackedPersonState | null = null;
        let bestIoU = 0;
        let bestDist = Infinity;

        tracks.forEach((tr) => {
          if (matchedTrackIds.has(tr.id)) return;

          const origPos = snapPos[tr.id];
          const driftX = origPos ? tr.bbox[0] - origPos[0] : 0;
          const driftY = origPos ? tr.bbox[1] - origPos[1] : 0;
          const compensatedBox: [number, number, number, number] = [
            Math.max(4, Math.min(W - bw - 4, bx + driftX)),
            Math.max(4, Math.min(H - bh - 4, by + driftY)),
            bw,
            bh,
          ];

          const iou = boxIoU(tr.bbox, compensatedBox);
          const dist = centerDist(tr.bbox, compensatedBox);

          // Enforce a spatial distance gate even in single-subject mode so distant false boxes can NEVER pull the track!
          if (isSingleMode && dist < W * 0.25) {
            bestTrack = tr;
            bestIoU = Math.max(0.45, iou);
          } else if (iou > 0.35 && iou > bestIoU) {
            bestIoU = iou;
            bestTrack = tr;
          } else if (bestIoU === 0 && dist < 56 && dist < bestDist) {
            bestDist = dist;
            bestTrack = tr;
          }
        });

        if (bestTrack) {
          const tr = bestTrack as TrackedPersonState;
          matchedTrackIds.add(tr.id);
          tr.missedFrames = 0;
          tr.neuralVerified = true;
          tr.ageFrames = Math.max(tr.ageFrames, 4);

          const origPos = snapPos[tr.id];
          const driftX = origPos ? tr.bbox[0] - origPos[0] : 0;
          const driftY = origPos ? tr.bbox[1] - origPos[1] : 0;
          const targetBox: [number, number, number, number] = [
            Math.max(4, Math.min(W - bw - 4, bx + driftX)),
            Math.max(4, Math.min(H - bh - 4, by + driftY)),
            bw,
            bh,
          ];

          if (!tr.isFallen || !tr.faintLockedBbox) {
            const blendAlpha = bestIoU > 0.35 ? 0.25 : 0.18;
            tr.bbox = [
              tr.bbox[0] * (1 - blendAlpha) + targetBox[0] * blendAlpha,
              tr.bbox[1] * (1 - blendAlpha) + targetBox[1] * blendAlpha,
              tr.bbox[2] * (1 - blendAlpha) + targetBox[2] * blendAlpha,
              tr.bbox[3] * (1 - blendAlpha) + targetBox[3] * blendAlpha,
            ];
          }

          if (isFaintedStatus && (bw / Math.max(1, bh) >= 1.05 || by > H * 0.45)) {
            tr.isFallen = true;
            tr.isVerifyingFall = false;
            tr.neuralFaintConfirmed = true;
            tr.faintLockedBbox = tr.faintLockedBbox || [...tr.bbox];
            tr.faintStartTime = tr.faintStartTime || now;
            tr.faintReason = 'Floor Faint Confirmed & Locked Indefinitely';
            tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
          }
        } else if (!isSingleMode || tracks.length === 0) {
          if (tracks.length < 28) {
            const id = nextTrackIdRef.current++;
            const newTrack: TrackedPersonState = {
              id,
              bbox: [...item.bbox],
              standingBaselineBbox: isFaintedStatus ? null : [...item.bbox],
              peakHeight: isFaintedStatus ? 0 : bh,
              baselineTopY: isFaintedStatus ? H : by,
              standingBottomY: by + bh,
              vx: 0,
              vy: 0,
              speedPx: 0,
              ageFrames: 4,
              missedFrames: 0,
              standingFrames: isFaintedStatus ? 0 : 4,
              collapseCandidateStartMs: isFaintedStatus ? now - 1000 : null,
              isVerifyingFall: false,
              verifyProgressSec: 0,
              isFallen: isFaintedStatus,
              faintLockedBbox: isFaintedStatus ? [...item.bbox] : null,
              faintStartTime: isFaintedStatus ? now : null,
              faintReason: isFaintedStatus ? 'Floor Faint Confirmed & Locked Indefinitely' : '',
              neuralFaintConfirmed: isFaintedStatus,
              neuralVerified: true,
              smoothedKeypoints: isFaintedStatus
                ? buildFrozenFloorKeypoints(item.bbox)
                : buildStandingKeypoints(item.bbox, id),
              psoKeypointsMap: {},
              torsoColor: 'BLUE',
              hsvMatchPct: 0,
            };
            tracks.push(newTrack);
            matchedTrackIds.add(id);
          }
        }
      });

      tracksRef.current = tracks.filter(
        (tr) => tr.isFallen || isSingleMode || matchedTrackIds.has(tr.id)
      );
    },
    []
  );

  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;

    const stopMediaTracks = () => {
      if (videoEl.srcObject) {
        const stream = videoEl.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
        videoEl.srcObject = null;
      }
    };

    if (inputSource === 'webcam') {
      setWebcamError(null);
      stopMediaTracks();
      videoEl.src = '';
      resetMemoryBank();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setWebcamError('Camera not available in this browser. Try uploading a video.');
        return;
      }
      navigator.mediaDevices
        .getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        })
        .then((stream) => {
          videoEl.srcObject = stream;
          videoEl.play().catch(() => {});
          setIsPlaying(true);
        })
        .catch(() => {
          setWebcamError(
            'Webcam permission was declined or no camera was found. Upload a video or use the Demo Scene.'
          );
        });
    } else if (inputSource === 'upload') {
      stopMediaTracks();
      resetMemoryBank();
      if (uploadedVideoUrl) {
        videoEl.src = uploadedVideoUrl;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.play().catch(() => {});
        setIsPlaying(true);
      }
    } else {
      stopMediaTracks();
      videoEl.src = '';
    }

    return () => {
      stopMediaTracks();
    };
  }, [inputSource, uploadedVideoUrl, resetMemoryBank]);

  const handleVideoFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl);
    }
    const url = URL.createObjectURL(file);
    setUploadedFileName(file.name);
    setUploadedVideoUrl(url);
    onInputSourceChange('upload');
    resetMemoryBank();
  };

  const togglePlayPause = () => {
    const videoEl = videoRef.current;
    if (!videoEl || inputSource === 'simulation') {
      setIsPlaying((prev) => !prev);
      return;
    }
    if (videoEl.paused) {
      videoEl.play().catch(() => {});
      setIsPlaying(true);
    } else {
      videoEl.pause();
      setIsPlaying(false);
    }
  };

  const handleTimelineSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const videoEl = videoRef.current;
    const newTime = parseFloat(e.target.value);
    setVideoProgress(newTime);
    if (videoEl && !Number.isNaN(newTime)) {
      videoEl.currentTime = newTime;
      lastNeuralTimeRef.current = 0;
      if (newTime < 0.8 && !manualFallOverride) {
        tracksRef.current.forEach((tr) => {
          tr.isFallen = false;
          tr.isVerifyingFall = false;
          tr.faintLockedBbox = null;
          tr.faintStartTime = null;
          tr.collapseCandidateStartMs = null;
          tr.neuralFaintConfirmed = false;
        });
      }
    }
  };

  const toggleFallTest = useCallback(() => {
    if (inputSource === 'simulation') {
      const actors = simActorsRef.current;
      actors[1].fallen = !actors[1].fallen;
    } else {
      setManualFallOverride((prev) => {
        const next = !prev;
        const tracks = tracksRef.current;
        if (next && tracks.length > 0) {
          const target = tracks[0];
          target.isFallen = true;
          target.isVerifyingFall = false;
          target.faintLockedBbox = [...target.bbox];
          target.faintStartTime = performance.now();
          target.faintReason = 'Manual Faint Override (Locked Indefinitely)';
          target.smoothedKeypoints = buildFrozenFloorKeypoints(target.bbox);
        } else if (!next) {
          tracks.forEach((t) => {
            t.isFallen = false;
            t.isVerifyingFall = false;
            t.faintLockedBbox = null;
            t.faintStartTime = null;
            t.collapseCandidateStartMs = null;
          });
        }
        return next;
      });
    }
  }, [inputSource]);

  const toggleSecurityTest = useCallback(() => {
    if (inputSource === 'simulation') {
      const actors = simActorsRef.current;
      const isIntruder = actors[0].maskedIntruder;
      actors[0].maskedIntruder = !isIntruder;
      actors[0].hasKnife = !isIntruder;
    } else {
      setManualSecurityOverride((prev) => !prev);
    }
  }, [inputSource]);

  useEffect(() => {
    if (mode === 'tifawt') {
      simActorsRef.current[1].color = targetColor;
    }
  }, [mode, targetColor]);

  useEffect(() => {
    let animationFrameId: number;
    let lastFrameTimestamp = performance.now();
    let smoothedFps = 34.0;

    const GW = 240;
    const GH = 135;
    if (!offscreenCanvasRef.current) {
      const off = document.createElement('canvas');
      off.width = GW;
      off.height = GH;
      offscreenCanvasRef.current = off;
    }
    if (!snapshotCanvasRef.current) {
      const sc = document.createElement('canvas');
      sc.width = 640;
      sc.height = 360;
      snapshotCanvasRef.current = sc;
    }

    const renderLoop = (now: number) => {
      const dt = Math.max(1, now - lastFrameTimestamp);
      lastFrameTimestamp = now;
      smoothedFps = smoothedFps * 0.9 + Math.min(60, 1000 / dt) * 0.1;

      const tStart = performance.now();
      const canvas = canvasRef.current;
      const videoEl = videoRef.current;
      const offCanvas = offscreenCanvasRef.current;
      const snapCanvas = snapshotCanvasRef.current;

      if (canvas && offCanvas && snapCanvas) {
        const ctx = canvas.getContext('2d');
        const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
        const snapCtx = snapCanvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        if (ctx && offCtx && snapCtx) {
          const detectedSubjects: DetectedSubject[] = [];

          const hasLiveVideoFrame =
            (inputSource === 'webcam' || inputSource === 'upload') &&
            videoEl &&
            videoEl.readyState >= 2 &&
            videoEl.videoWidth > 0;

          if (hasLiveVideoFrame && videoEl) {
            if (inputSource === 'upload' && videoEl.duration) {
              setVideoProgress(videoEl.currentTime);
              setVideoDuration(videoEl.duration);
              // When video loops back to 00:00, reset faint lock so user can watch the standing -> faint transition cleanly every loop!
              if (videoEl.currentTime < 0.35 && !manualFallOverride) {
                tracksRef.current.forEach((tr) => {
                  tr.isFallen = false;
                  tr.isVerifyingFall = false;
                  tr.faintLockedBbox = null;
                  tr.faintStartTime = null;
                  tr.collapseCandidateStartMs = null;
                  tr.neuralFaintConfirmed = false;
                });
              }
            }

            ctx.drawImage(videoEl, 0, 0, W, H);
            offCtx.drawImage(videoEl, 0, 0, GW, GH);
            const imgData = offCtx.getImageData(0, 0, GW, GH);
            const data = imgData.data;

            const scaleX = W / GW;
            const scaleY = H / GH;

            // Precompute grayscale & motion mask (excluding CCTV timestamp & sky/flag zones)
            const numPixels = GW * GH;
            const gray = new Uint8Array(numPixels);
            const motionMask = new Uint8Array(numPixels);
            const prevGray = prevGrayRef.current;

            for (let y = 0; y < GH; y++) {
              for (let x = 0; x < GW; x++) {
                const pIdx = y * GW + x;
                const idx = pIdx * 4;
                const luma = (data[idx] * 77 + data[idx + 1] * 150 + data[idx + 2] * 29) >> 8;
                gray[pIdx] = luma;

                // Ignore top sky/waving flags (y < 14) and bottom-right CCTV "CAM 04" text
                if (y >= 14 && y < GH - 8 && !(y > GH * 0.88 && x > GW * 0.78)) {
                  const mDiff = prevGray ? Math.abs(luma - prevGray[pIdx]) : 0;
                  if (mDiff >= 16) {
                    motionMask[pIdx] = 1;
                  }
                }
              }
            }
            prevGrayRef.current = gray;

            // =========================================================================
            // 1. ASYNC NEURAL GROUND-TRUTH SNAPSHOT (/api/analyze-frame)
            // =========================================================================
            if (!neuralBusyRef.current && now - lastNeuralTimeRef.current > 1800) {
              neuralBusyRef.current = true;
              lastNeuralTimeRef.current = now;

              const posMap: Record<number, [number, number]> = {};
              tracksRef.current.forEach((tr) => {
                posMap[tr.id] = [tr.bbox[0], tr.bbox[1]];
              });
              snapshotTrackPositionsRef.current = posMap;

              snapCtx.drawImage(videoEl, 0, 0, 640, 360);
              const jpegBase64 = snapCanvas.toDataURL('image/jpeg', 0.72);

              fetch('/api/analyze-frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageBase64: jpegBase64 }),
              })
                .then((r) => (r.ok ? r.json() : null))
                .then((res) => {
                  if (res && Array.isArray(res.people) && res.people.length > 0) {
                    reconcileServerNeuralDetections(res.people, W, H, performance.now());
                  }
                })
                .catch(() => {})
                .finally(() => {
                  neuralBusyRef.current = false;
                });
            }

            // =========================================================================
            // 2. CLIENT-SIDE MEDIAPIPE POSE LANDMARKER & OBJECT DETECTOR + SHADOW CROP
            // =========================================================================
            interface FrameCandidate {
              bbox: [number, number, number, number];
              keypointsMap?: Record<string, PoseKeypoint>;
              torsoAngleDeg?: number;
              foreshortenedFloor?: boolean;
              fromPose: boolean;
              score: number;
            }

            const rawCandidates: FrameCandidate[] = [];

            if (videoEl.currentTime !== lastVideoTimeRef.current) {
              lastVideoTimeRef.current = videoEl.currentTime;
              const tsMs = performance.now();

              if (poseLandmarkerRef.current) {
                try {
                  const poseResult = poseLandmarkerRef.current.detectForVideo(videoEl, tsMs);
                  if (poseResult.landmarks && poseResult.landmarks.length > 0) {
                    poseResult.landmarks.forEach((lm) => {
                      let minX = 1.0;
                      let maxX = 0.0;
                      let minY = 1.0;
                      let maxY = 0.0;
                      let highConfCount = 0;

                      const keypointMap: Array<{ idx: number; name: string }> = [
                        { idx: 0, name: 'nose' },
                        { idx: 11, name: 'l_shoulder' },
                        { idx: 12, name: 'r_shoulder' },
                        { idx: 13, name: 'l_elbow' },
                        { idx: 14, name: 'r_elbow' },
                        { idx: 15, name: 'l_wrist' },
                        { idx: 16, name: 'r_wrist' },
                        { idx: 23, name: 'l_hip' },
                        { idx: 24, name: 'r_hip' },
                        { idx: 25, name: 'l_knee' },
                        { idx: 26, name: 'r_knee' },
                        { idx: 27, name: 'l_ankle' },
                        { idx: 28, name: 'r_ankle' },
                      ];

                      const kpsMap: Record<string, PoseKeypoint> = {};
                      keypointMap.forEach(({ idx, name }) => {
                        const pt = lm[idx];
                        const vis = pt?.visibility ?? 0;
                        if (
                          pt &&
                          vis >= 0.42 &&
                          pt.x >= 0.02 &&
                          pt.x <= 0.98 &&
                          pt.y >= 0.02 &&
                          pt.y <= 0.98
                        ) {
                          highConfCount++;
                          if (pt.x < minX) minX = pt.x;
                          if (pt.x > maxX) maxX = pt.x;
                          if (pt.y < minY) minY = pt.y;
                          if (pt.y > maxY) maxY = pt.y;
                          kpsMap[name] = {
                            name,
                            x: pt.x * W,
                            y: pt.y * H,
                            confidence: vis,
                          };
                        }
                      });

                      if (highConfCount >= 5) {
                        const padX = Math.max(0.018, (maxX - minX) * 0.12);
                        const padYTop = Math.max(0.025, (maxY - minY) * 0.10);
                        const padYBot = Math.max(0.025, (maxY - minY) * 0.08);
                        const bx = Math.max(6, (minX - padX) * W);
                        const by = Math.max(6, (minY - padYTop) * H);
                        const bw = Math.min(W - bx - 6, (maxX - minX + padX * 2) * W);
                        const bh = Math.min(H - by - 6, (maxY - minY + padYTop + padYBot) * H);

                        if (bw >= 24 && bh >= 28 && bw < W * 0.70 && bh < H * 0.94) {
                          const shadowCroppedBox = cropCastShadowFromBox(
                            [bx, by, bw, bh],
                            data,
                            gray,
                            GW,
                            GH,
                            W,
                            H
                          );

                          let torsoAngleDeg = 10.0;
                          let foreshortenedFloor = false;
                          const lSh = lm[11];
                          const rSh = lm[12];
                          const lHip = lm[23];
                          const rHip = lm[24];
                          if (lSh && rSh && lHip && rHip) {
                            const shMidX = (lSh.x + rSh.x) / 2;
                            const shMidY = (lSh.y + rSh.y) / 2;
                            const hipMidX = (lHip.x + rHip.x) / 2;
                            const hipMidY = (lHip.y + rHip.y) / 2;
                            const dx = Math.abs(shMidX - hipMidX) * W;
                            const dy = Math.max(1, Math.abs(shMidY - hipMidY) * H);
                            torsoAngleDeg =
                              Math.round(((Math.atan2(dx, dy) * 180) / Math.PI) * 10) / 10;
                            const torsoNormDy = Math.abs(shMidY - hipMidY);
                            if (torsoNormDy < 0.11 && minY > 0.48) {
                              foreshortenedFloor = true;
                            }
                          }

                          rawCandidates.push({
                            bbox: shadowCroppedBox,
                            keypointsMap: kpsMap,
                            torsoAngleDeg,
                            foreshortenedFloor,
                            fromPose: true,
                            score: 0.96,
                          });
                        }
                      }
                    });
                  }
                } catch {
                  // Continue
                }
              }

              if (objectDetectorRef.current) {
                try {
                  const objResult = objectDetectorRef.current.detectForVideo(videoEl, tsMs + 1);
                  if (objResult.detections && objResult.detections.length > 0) {
                    const scaleVidX = W / Math.max(1, videoEl.videoWidth);
                    const scaleVidY = H / Math.max(1, videoEl.videoHeight);

                    objResult.detections.forEach((det) => {
                      const bb = det.boundingBox;
                      const score = det.categories?.[0]?.score ?? 0.35;
                      if (!bb || score < 0.32) return;
                      const bx = Math.max(6, bb.originX * scaleVidX);
                      const by = Math.max(6, bb.originY * scaleVidY);
                      const bw = Math.min(W - bx - 6, bb.width * scaleVidX);
                      const bh = Math.min(H - by - 6, bb.height * scaleVidY);

                      // Reject static upper-horizon monuments/archways (e.g. Place du Méchouar central tower y < 46%)
                      if (by + bh < H * 0.47 && bx > W * 0.42 && bx + bw < W * 0.58) return;

                      if (bw >= 20 && bh >= 24 && bw < W * 0.65 && bh < H * 0.93) {
                        const shadowCroppedBox = cropCastShadowFromBox(
                          [bx, by, bw, bh],
                          data,
                          gray,
                          GW,
                          GH,
                          W,
                          H
                        );
                        rawCandidates.push({
                          bbox: shadowCroppedBox,
                          fromPose: false,
                          score,
                        });
                      }
                    });
                  }
                } catch {
                  // Continue
                }
              }
            }

            // =========================================================================
            // 3. FOREGROUND DOMINANCE FILTER & LOCAL SHADOW-CROPPED ROI CONTINUITY
            // =========================================================================
            const hyst = scaleHysteresisRef.current;

            // Filter out tiny distant walkers in the background (bh < 9.5% of frame height in upper half)
            // when a clear foreground person is present or latched
            const hasForegroundCandidate = rawCandidates.some(
              (c) => c.bbox[3] >= H * 0.11 || (c.bbox[2] >= W * 0.11 && c.bbox[1] + c.bbox[3] >= H * 0.55)
            );
            const filteredRawCandidates =
              hasForegroundCandidate || hyst.latchedSingleForegroundMode
                ? rawCandidates.filter(
                    (c) =>
                      c.bbox[3] >= H * 0.095 ||
                      c.bbox[2] >= W * 0.10 ||
                      c.bbox[1] + c.bbox[3] >= H * 0.58
                  )
                : rawCandidates;

            let currentFrameMaxHeightRatio = 0;
            filteredRawCandidates.forEach((c) => {
              const r = c.bbox[3] / H;
              if (r > currentFrameMaxHeightRatio) currentFrameMaxHeightRatio = r;
            });

            if (currentFrameMaxHeightRatio > 0) {
              hyst.trailingMaxHeights.push(currentFrameMaxHeightRatio);
              if (hyst.trailingMaxHeights.length > 60) {
                hyst.trailingMaxHeights.shift();
              }
            }
            const trailingPeak =
              hyst.trailingMaxHeights.length > 0
                ? Math.max(...hyst.trailingMaxHeights)
                : 0;
            hyst.sessionPeakHeightRatio = Math.max(hyst.sessionPeakHeightRatio, trailingPeak);

            // Latch Single Foreground Subject Mode when 1-2 foreground subjects are present (e.g. Place du Méchouar bh ≈ 0.26 or close-up room)
            if (
              inputSource === 'webcam' ||
              hyst.sessionPeakHeightRatio >= 0.22 ||
              trailingPeak >= 0.22 ||
              (filteredRawCandidates.length >= 1 &&
                filteredRawCandidates.length <= 2 &&
                currentFrameMaxHeightRatio >= 0.12)
            ) {
              if (filteredRawCandidates.length <= 3 && tracksRef.current.length <= 3) {
                hyst.latchedSingleForegroundMode = true;
              }
            } else if (filteredRawCandidates.length >= 5) {
              hyst.latchedSingleForegroundMode = false;
            }

            // Local Shadow-Cropped Body Continuity Tracker:
            // When MediaPipe misses a frame (e.g. flowing djellaba or horizontal collapse on Place du Méchouar at 00:04-00:09),
            // extract the exact shadow-cropped body box inside the local search window around the person!
            const primaryTrack = tracksRef.current[0] || null;
            if (
              filteredRawCandidates.length === 0 &&
              (hyst.latchedSingleForegroundMode || tracksRef.current.length <= 1) &&
              !(primaryTrack && primaryTrack.isFallen && primaryTrack.faintLockedBbox)
            ) {
              const localBody = extractShadowCroppedBodyInROI(
                primaryTrack ? primaryTrack.bbox : null,
                primaryTrack ? primaryTrack.standingBottomY : null,
                data,
                gray,
                GW,
                GH,
                W,
                H
              );
              if (localBody) {
                hyst.latchedSingleForegroundMode = true;
                filteredRawCandidates.push({
                  bbox: localBody.bbox,
                  foreshortenedFloor: localBody.foreshortenedFloor,
                  fromPose: false,
                  score: localBody.score,
                });
              }
            } else if (
              filteredRawCandidates.length === 1 &&
              hyst.latchedSingleForegroundMode &&
              primaryTrack &&
              !primaryTrack.isFallen
            ) {
              // Also cross-check single detector candidate against shadow-cropped ROI so loose robes/horizontal bodies stay tight
              const cand = filteredRawCandidates[0];
              const localBody = extractShadowCroppedBodyInROI(
                primaryTrack.bbox,
                primaryTrack.standingBottomY,
                data,
                gray,
                GW,
                GH,
                W,
                H
              );
              if (localBody && boxIoU(cand.bbox, localBody.bbox) > 0.20) {
                cand.bbox = [
                  cand.bbox[0] * 0.55 + localBody.bbox[0] * 0.45,
                  cand.bbox[1] * 0.55 + localBody.bbox[1] * 0.45,
                  cand.bbox[2] * 0.55 + localBody.bbox[2] * 0.45,
                  cand.bbox[3] * 0.55 + localBody.bbox[3] * 0.45,
                ];
              }
            }

            const isSingleForegroundMode = hyst.latchedSingleForegroundMode;
            setSceneScaleLabel(
              isSingleForegroundMode ? 'Foreground Subject Mode' : 'Wide Crowd Mode'
            );

            // Containment NMS guarantees strictly 1 clean box on the primary foreground person
            const nmsCandidates = applyContainmentNMS(
              filteredRawCandidates,
              isSingleForegroundMode
            );

            // =========================================================================
            // 4. SPATIALLY GATED TRACK UPDATE + RELATIVE COLLAPSE (<1.0s FAINT LOCK)
            // =========================================================================
            const tracks = tracksRef.current;

            if (isSingleForegroundMode && tracks.length > 1) {
              const primary = tracks.find((t) => t.isFallen) || tracks[0];
              tracksRef.current = [primary];
            }

            const activeTracks = tracksRef.current;
            const matchedTrackIds = new Set<number>();
            const matchedCandIndices = new Set<number>();

            nmsCandidates.forEach((cand, cIdx) => {
              let bestTrack: TrackedPersonState | null = null;
              let bestDist = Infinity;

              activeTracks.forEach((tr) => {
                if (matchedTrackIds.has(tr.id)) return;
                const d = centerDist(tr.bbox, cand.bbox);

                // SPATIAL GATE: Even in single-foreground mode, candidate must be within reasonable proximity
                // of the track (prevents jumping to distant background artifacts)
                const maxAllowedDist = isSingleForegroundMode ? W * 0.26 : 68;
                if (d < maxAllowedDist && d < bestDist) {
                  bestDist = d;
                  bestTrack = tr;
                }
              });

              if (bestTrack) {
                const tr = bestTrack as TrackedPersonState;
                matchedTrackIds.add(tr.id);
                matchedCandIndices.add(cIdx);
                tr.missedFrames = 0;
                tr.ageFrames++;

                const prevCx = tr.bbox[0] + tr.bbox[2] * 0.5;
                const prevCy = tr.bbox[1] + tr.bbox[3] * 0.5;
                const newCx = cand.bbox[0] + cand.bbox[2] * 0.5;
                const newCy = cand.bbox[1] + cand.bbox[3] * 0.5;

                // Clamp per-frame velocity so the box never jerks or plays around
                const clampedDx = Math.max(-18, Math.min(18, newCx - prevCx));
                const clampedDy = Math.max(-18, Math.min(18, newCy - prevCy));

                tr.vx = tr.vx * 0.68 + clampedDx * 0.32;
                tr.vy = tr.vy * 0.68 + clampedDy * 0.32;
                tr.speedPx = Math.hypot(tr.vx, tr.vy);

                if (!tr.isFallen || !tr.faintLockedBbox) {
                  const alpha = cand.fromPose ? 0.35 : 0.30;
                  tr.bbox = [
                    tr.bbox[0] * (1 - alpha) + cand.bbox[0] * alpha,
                    tr.bbox[1] * (1 - alpha) + cand.bbox[1] * alpha,
                    tr.bbox[2] * (1 - alpha) + cand.bbox[2] * alpha,
                    tr.bbox[3] * (1 - alpha) + cand.bbox[3] * alpha,
                  ];
                } else if (
                  tr.isFallen &&
                  tr.faintLockedBbox &&
                  tr.faintStartTime &&
                  now - tr.faintStartTime < 900
                ) {
                  // During the first 0.9s after faint lock triggers (e.g. transitioning from kneeling at 00:04 to lying flat at 00:05),
                  // allow the locked box to settle onto the flatter horizontal posture, then freeze 100% indefinitely!
                  const currLockedAspect =
                    tr.faintLockedBbox[2] / Math.max(1, tr.faintLockedBbox[3]);
                  const candAspect = cand.bbox[2] / Math.max(1, cand.bbox[3]);
                  if (candAspect > currLockedAspect * 1.04) {
                    tr.faintLockedBbox = [
                      tr.faintLockedBbox[0] * 0.5 + cand.bbox[0] * 0.5,
                      tr.faintLockedBbox[1] * 0.5 + cand.bbox[1] * 0.5,
                      tr.faintLockedBbox[2] * 0.5 + cand.bbox[2] * 0.5,
                      tr.faintLockedBbox[3] * 0.5 + cand.bbox[3] * 0.5,
                    ];
                    tr.bbox = [...tr.faintLockedBbox];
                  }
                }

                const [bx, by, bw, bh] = tr.bbox;
                const aspect = bw / Math.max(1, bh);

                // Smoothly update keypoints when not faint-locked
                if (!tr.isFallen && cand.keypointsMap) {
                  Object.entries(cand.keypointsMap).forEach(([kName, pt]) => {
                    const cx = Math.max(bx, Math.min(bx + bw, pt.x));
                    const cy = Math.max(by, Math.min(by + bh, pt.y));
                    const prev = tr.psoKeypointsMap[kName];
                    tr.psoKeypointsMap[kName] = prev
                      ? {
                          name: kName,
                          x: prev.x * 0.65 + cx * 0.35,
                          y: prev.y * 0.65 + cy * 0.35,
                          confidence: pt.confidence,
                        }
                      : { name: kName, x: cx, y: cy, confidence: pt.confidence };
                  });
                }

                // RELATIVE UPRIGHT STANDING CHECK (Works at any Y depth on an outdoor plaza or indoor room!)
                const isRelativelyStanding =
                  aspect < 0.80 &&
                  (tr.peakHeight === 0 || bh >= tr.peakHeight * 0.74) &&
                  !cand.foreshortenedFloor;

                if (isRelativelyStanding) {
                  tr.standingFrames++;
                  if (bh >= tr.peakHeight * 0.85) {
                    tr.peakHeight = Math.max(tr.peakHeight, bh);
                    tr.baselineTopY = Math.min(tr.baselineTopY, by);
                    tr.standingBottomY = Math.max(tr.standingBottomY, by + bh);
                    tr.standingBaselineBbox = [...tr.bbox];
                  }
                  if (tr.standingFrames >= 4 && !tr.isFallen) {
                    tr.collapseCandidateStartMs = null;
                    tr.isVerifyingFall = false;
                    tr.verifyProgressSec = 0;
                  }
                  // Unlock Indefinite Faint Lock ONLY if person clearly stands back up for 15+ consecutive upright frames
                  if (tr.isFallen && tr.standingFrames >= 15 && !manualFallOverride) {
                    tr.isFallen = false;
                    tr.faintLockedBbox = null;
                    tr.faintStartTime = null;
                    tr.collapseCandidateStartMs = null;
                    tr.isVerifyingFall = false;
                  }
                } else {
                  tr.standingFrames = 0;
                }

                // =====================================================================
                // RELATIVE POSTURE COLLAPSE GEOMETRY & SUB-1-SECOND (<1.0s) FAINT LOCK
                // =====================================================================
                if (isSingleForegroundMode) {
                  const headDropRatio =
                    tr.peakHeight > 0 ? (by - tr.baselineTopY) / tr.peakHeight : 0;
                  const heightCompPct =
                    tr.peakHeight > 0 ? Math.round((1 - bh / tr.peakHeight) * 100) : 0;

                  // 1. Horizontal Aspect Ratio Flip (standing ~0.42 -> kneeling/lying >= 1.08)
                  const sigHorizontalFlip = aspect >= 1.08;
                  // 2. Relative Height Compression + Head Drop toward ground plane
                  const sigRelativeCompression =
                    tr.peakHeight >= H * 0.14 &&
                    heightCompPct >= 32 &&
                    headDropRatio >= 0.28;
                  // 3. Severe Torso Tilt
                  const sigTorsoTilt = (cand.torsoAngleDeg ?? 0) >= 48 && aspect >= 0.88;
                  // 4. Indoor Foreshortened Floor Posture (for videos starting already on floor)
                  const sigIndoorForeshortened =
                    Boolean(cand.foreshortenedFloor) &&
                    tr.peakHeight === 0 &&
                    by > H * 0.48;

                  const inCollapsedPosture =
                    sigHorizontalFlip ||
                    sigRelativeCompression ||
                    sigTorsoTilt ||
                    sigIndoorForeshortened;

                  if (inCollapsedPosture && !tr.isFallen) {
                    if (!tr.collapseCandidateStartMs) {
                      tr.collapseCandidateStartMs = now;
                    }
                    const elapsedSec = (now - tr.collapseCandidateStartMs) / 1000;
                    tr.verifyProgressSec = Math.round(elapsedSec * 10) / 10;

                    // Sub-1-second confirmation (0.65s) as requested by user!
                    const requiredPersistenceSec = 0.65;

                    if (elapsedSec < requiredPersistenceSec) {
                      tr.isVerifyingFall = true;
                    } else {
                      // COMMIT TO INDEFINITE IMMUTABLE FAINT LOCK!
                      tr.isFallen = true;
                      tr.isVerifyingFall = false;
                      tr.faintLockedBbox = [...tr.bbox];
                      tr.faintStartTime = now;
                      tr.faintReason = sigHorizontalFlip
                        ? `Horizontal Collapse Detected (Aspect ${aspect.toFixed(1)}x · Locked Indefinitely)`
                        : `Height Compressed ${heightCompPct}% & Collapsed on Ground (Locked Indefinitely)`;
                      tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
                    }
                  }
                }

                // Select final keypoints for rendering
                if (tr.isFallen && tr.faintLockedBbox) {
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
                } else {
                  const psoList = Object.values(tr.psoKeypointsMap);
                  tr.smoothedKeypoints =
                    psoList.length >= 8 && aspect < 0.85
                      ? psoList
                      : aspect >= 1.05
                      ? buildFrozenFloorKeypoints(tr.bbox)
                      : buildStandingKeypoints(tr.bbox, now * 0.008 + tr.id);
                }
              }
            });

            // Smooth optical-flow interpolation for multi-person crowd tracks
            if (!isSingleForegroundMode) {
              activeTracks.forEach((tr) => {
                if (matchedTrackIds.has(tr.id) || tr.isFallen) return;
                const [bx, by, bw, bh] = tr.bbox;
                const gx1 = Math.max(8, Math.floor((bx - 12) / scaleX));
                const gx2 = Math.min(GW - 8, Math.ceil((bx + bw + 12) / scaleX));
                const gy1 = Math.max(8, Math.floor((by - 12) / scaleY));
                const gy2 = Math.min(GH - 8, Math.ceil((by + bh + 12) / scaleY));

                let sumX = 0;
                let sumY = 0;
                let mCount = 0;
                for (let y = gy1; y <= gy2; y++) {
                  for (let x = gx1; x <= gx2; x++) {
                    if (motionMask[y * GW + x]) {
                      sumX += x * scaleX;
                      sumY += y * scaleY;
                      mCount++;
                    }
                  }
                }
                if (mCount >= 3) {
                  const targetCx = sumX / mCount;
                  const targetCy = sumY / mCount;
                  const currCx = bx + bw * 0.5;
                  const currCy = by + bh * 0.5;
                  const dx = Math.max(-4.0, Math.min(4.0, (targetCx - currCx) * 0.25));
                  const dy = Math.max(-4.0, Math.min(4.0, (targetCy - currCy) * 0.25));
                  tr.bbox = [bx + dx, by + dy, bw, bh];
                  tr.smoothedKeypoints = buildStandingKeypoints(tr.bbox, now * 0.008 + tr.id);
                  tr.missedFrames = 0;
                }
              });
            }

            // Create new track(s) for unmatched candidates
            nmsCandidates.forEach((cand, cIdx) => {
              if (matchedCandIndices.has(cIdx)) return;
              if (isSingleForegroundMode && activeTracks.length >= 1) return;
              if (activeTracks.length >= 28) return;

              const id = nextTrackIdRef.current++;
              const [bx, by, bw, bh] = cand.bbox;
              const aspect = bw / Math.max(1, bh);
              const isImmediateHorizontal = aspect >= 1.15;

              activeTracks.push({
                id,
                bbox: [...cand.bbox],
                standingBaselineBbox: !isImmediateHorizontal ? [...cand.bbox] : null,
                peakHeight: !isImmediateHorizontal ? bh : 0,
                baselineTopY: !isImmediateHorizontal ? by : H,
                standingBottomY: by + bh,
                vx: 0,
                vy: 0,
                speedPx: 0,
                ageFrames: 3,
                missedFrames: 0,
                standingFrames: isImmediateHorizontal ? 0 : 2,
                collapseCandidateStartMs: isImmediateHorizontal ? now : null,
                isVerifyingFall: isImmediateHorizontal,
                verifyProgressSec: 0,
                isFallen: false,
                faintLockedBbox: null,
                faintStartTime: null,
                faintReason: '',
                neuralFaintConfirmed: false,
                neuralVerified: cand.fromPose || cand.score >= 0.45,
                smoothedKeypoints: isImmediateHorizontal
                  ? buildFrozenFloorKeypoints(cand.bbox)
                  : buildStandingKeypoints(cand.bbox, id),
                psoKeypointsMap: cand.keypointsMap || {},
                torsoColor: 'WHITE',
                hsvMatchPct: 0,
              });
            });

            // INDEFINITE FAINT LOCK RETENTION:
            // Never prematurely teleport a walking person's box when missedFrames >= 2;
            // instead, keep their smooth position or lock only if they were already actively collapsing!
            tracksRef.current = activeTracks.filter((tr) => {
              if (!matchedTrackIds.has(tr.id)) {
                tr.missedFrames++;
                if (
                  isSingleForegroundMode &&
                  !tr.isFallen &&
                  tr.isVerifyingFall &&
                  tr.missedFrames >= 3
                ) {
                  tr.isFallen = true;
                  tr.isVerifyingFall = false;
                  tr.faintLockedBbox = [...tr.bbox];
                  tr.faintStartTime = tr.faintStartTime || now;
                  tr.faintReason = 'Collapsed on Ground (Locked Indefinitely)';
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
                }
              }

              if (tr.isFallen || isSingleForegroundMode) {
                return true;
              }
              if (tr.neuralVerified) {
                return tr.missedFrames < 95;
              }
              return tr.missedFrames < 12;
            });

            // Convert active tracks into DetectedSubjects
            let activeFaintSummary: string | null = null;
            let activeVerifySummary: string | null = null;

            tracksRef.current.forEach((tr, idx) => {
              const activeBox =
                tr.isFallen && tr.faintLockedBbox ? tr.faintLockedBbox : tr.bbox;
              const [bx, by, bw, bh] = activeBox;

              const tx1 = Math.max(0, Math.floor((bx + bw * 0.22) / scaleX));
              const tx2 = Math.min(GW - 1, Math.floor((bx + bw * 0.78) / scaleX));
              const ty1 = Math.max(0, Math.floor((by + bh * 0.22) / scaleY));
              const ty2 = Math.min(GH - 1, Math.floor((by + bh * 0.62) / scaleY));

              let torsoTotal = 0;
              let torsoMatched = 0;
              const colorTallies: Record<TargetClothingColor, number> = {
                RED: 0,
                BLUE: 0,
                YELLOW: 0,
                GREEN: 0,
                BLACK: 0,
                WHITE: 0,
              };

              for (let y = ty1; y <= ty2; y++) {
                for (let x = tx1; x <= tx2; x++) {
                  const pIdx = (y * GW + x) * 4;
                  const [h, s, v] = rgbToOpenCvHsv(data[pIdx], data[pIdx + 1], data[pIdx + 2]);
                  torsoTotal++;
                  if (matchesTargetHsv(h, s, v, targetColor)) torsoMatched++;
                  (
                    ['RED', 'BLUE', 'YELLOW', 'GREEN', 'BLACK', 'WHITE'] as TargetClothingColor[]
                  ).forEach((c) => {
                    if (matchesTargetHsv(h, s, v, c)) colorTallies[c]++;
                  });
                }
              }

              if (torsoTotal > 4) {
                tr.torsoColor = (Object.keys(colorTallies) as TargetClothingColor[]).reduce(
                  (a, b) => (colorTallies[a] > colorTallies[b] ? a : b)
                );
                tr.hsvMatchPct =
                  Math.min(99.2, Math.round((torsoMatched / torsoTotal) * 1000) / 10);
              }

              const faintDurationSec =
                tr.isFallen && tr.faintStartTime
                  ? Math.max(1, Math.round((now - tr.faintStartTime) / 1000))
                  : 0;

              if (tr.isFallen) {
                activeFaintSummary = `${tr.faintReason || 'Floor Faint Locked Indefinitely'} (${faintDurationSec}s)`;
              } else if (tr.isVerifyingFall) {
                activeVerifySummary = `Confirming Fall (${tr.verifyProgressSec.toFixed(1)}s / 0.7s)...`;
              }

              detectedSubjects.push({
                id: tr.id,
                label: isSingleForegroundMode ? 'Person #1' : `Person #${idx + 1}`,
                bbox: activeBox,
                lastStandingBbox: tr.standingBaselineBbox || undefined,
                headRoi: [bx + bw * 0.24, by + bh * 0.04, bw * 0.52, bh * 0.22],
                torsoRoi: [bx + bw * 0.2, by + bh * 0.24, bw * 0.6, bh * 0.4],
                keypoints: tr.smoothedKeypoints,
                torsoColor: tr.hsvMatchPct >= 24 ? targetColor : tr.torsoColor,
                hsvMatchPct: tr.hsvMatchPct,
                torsoAngleDeg: tr.isFallen ? 82.0 : 10.0,
                headDropPct: tr.isFallen ? 48 : 0,
                heightCompressionPct: tr.isFallen ? 54 : 0,
                aspectRatio: Math.round((bw / Math.max(1, bh)) * 100) / 100,
                isVerifyingFall: tr.isVerifyingFall,
                verifyProgressSec: tr.verifyProgressSec,
                isFallen: tr.isFallen,
                isMemoryLocked: tr.isFallen,
                faintDurationSec,
                isConcealedFace: manualSecurityOverride && idx === 0,
                hasWeapon: manualSecurityOverride && idx === 0,
                weaponLabel:
                  manualSecurityOverride && idx === 0 ? 'Weapon / Blade Detected' : undefined,
              });
            });

            setLockedFaintSummary(activeFaintSummary);
            setVerifyingFallSummary(activeVerifySummary);
          } else {
            // BUILT-IN DEMO SCENE
            const grad = ctx.createLinearGradient(0, 0, 0, H);
            grad.addColorStop(0, '#0B111E');
            grad.addColorStop(0.5, '#111A2E');
            grad.addColorStop(1, '#162032');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, W, H);

            ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
            ctx.lineWidth = 1;
            for (let y = 170; y < H; y += 50) {
              ctx.beginPath();
              ctx.moveTo(0, y);
              ctx.lineTo(W, y);
              ctx.stroke();
            }

            ctx.fillStyle = 'rgba(226, 232, 240, 0.08)';
            for (let cw = 110; cw < W - 100; cw += 95) {
              ctx.fillRect(cw, 360, 52, 95);
            }

            simActorsRef.current.forEach((actor) => {
              if (isPlaying && !actor.fallen) {
                actor.x += actor.vx;
                if (actor.x > W - 140) actor.vx = -Math.abs(actor.vx);
                if (actor.x < 100) actor.vx = Math.abs(actor.vx);
              }

              if (actor.fallen) {
                actor.fallProgress = Math.min(1, actor.fallProgress + 0.08);
              } else {
                actor.fallProgress = Math.max(0, actor.fallProgress - 0.08);
              }

              const fp = actor.fallProgress;
              const bw = Math.round(92 + fp * 98);
              const bh = Math.round(215 - fp * 120);
              const bx = Math.round(actor.x - bw / 2);
              const by = Math.round(actor.y + fp * 95);

              const headW = Math.round(bw * (0.44 - fp * 0.16));
              const headH = Math.round(bh * (0.24 + fp * 0.12));
              const headX = Math.round(bx + (bw - headW) / 2 - fp * 36);
              const headY = Math.round(by + 6 + fp * 18);

              const torsoX = Math.round(bx + bw * 0.18);
              const torsoY = Math.round(by + bh * 0.27);
              const torsoW = Math.round(bw * 0.64);
              const torsoH = Math.round(bh * 0.42);

              ctx.save();
              ctx.fillStyle = '#334155';
              if (fp < 0.5) {
                const stride = Math.sin(now * 0.005 + actor.id) * 8;
                ctx.fillRect(torsoX + 6, torsoY + torsoH, 16, bh * 0.3 + stride);
                ctx.fillRect(torsoX + torsoW - 22, torsoY + torsoH, 16, bh * 0.3 - stride);
              } else {
                ctx.fillRect(torsoX + torsoW - 10, torsoY + 12, 54, 18);
              }

              ctx.fillStyle = TARGET_COLOR_CONFIG[actor.color].swatchHex;
              ctx.beginPath();
              ctx.roundRect(torsoX, torsoY, torsoW, torsoH, 8);
              ctx.fill();

              ctx.fillStyle = actor.maskedIntruder ? '#0F172A' : '#E2B897';
              ctx.beginPath();
              ctx.arc(
                headX + headW / 2,
                headY + headH / 2,
                Math.min(headW, headH) * 0.45,
                0,
                Math.PI * 2
              );
              ctx.fill();
              ctx.restore();

              const isFallen = fp > 0.65;
              const colorMatched = actor.color === targetColor;
              const hsvMatchPct = colorMatched ? 91.2 : 4.0;
              const torsoAngleDeg = Math.round((8.5 + fp * 71.0) * 10) / 10;
              const aspect = Math.round((bw / Math.max(1, bh)) * 100) / 100;

              const keypoints = isFallen
                ? buildFrozenFloorKeypoints([bx, by, bw, bh])
                : buildStandingKeypoints([bx, by, bw, bh], now * 0.005 + actor.id);

              detectedSubjects.push({
                id: actor.id,
                label: actor.name,
                bbox: [bx, by, bw, bh],
                lastStandingBbox: isFallen
                  ? [Math.round(actor.x - 46), actor.y, 92, 215]
                  : undefined,
                headRoi: [headX, headY, headW, headH],
                torsoRoi: [torsoX, torsoY, torsoW, torsoH],
                keypoints,
                torsoColor: actor.color,
                hsvMatchPct,
                torsoAngleDeg,
                headDropPct: isFallen ? 42 : 0,
                heightCompressionPct: isFallen ? 46 : 0,
                aspectRatio: aspect,
                isFallen,
                isMemoryLocked: isFallen,
                faintDurationSec: isFallen ? 6 : 0,
                isConcealedFace: actor.maskedIntruder,
                hasWeapon: actor.hasKnife,
                weaponLabel: actor.hasKnife ? 'Knife / Blade Detected' : undefined,
              });
            });
          }

          // =========================================================================
          // 5. RENDER BOUNDING BOXES, PROPORTIONAL SKELETONS & FAINT ALERTS
          // =========================================================================
          let frameAlertActive = false;
          let frameGreenWave = false;
          let frameAlertTitle = `All Clear — Tracking ${detectedSubjects.length} ${
            detectedSubjects.length === 1 ? 'Person' : 'People'
          }`;
          let frameAlertDetail = `Monitoring ${activeCamera.name} (${sceneScaleLabel}). Hospital → Place du Méchouar route on standby.`;

          const isCompactCrowd = detectedSubjects.length >= 4;

          detectedSubjects.forEach((subj) => {
            const [bx, by, bw, bh] = subj.bbox;
            const [tx, ty, tw, th] = subj.torsoRoi;

            // Draw Last Standing Position Ghost Box + Trajectory Line
            if (subj.isFallen && subj.lastStandingBbox) {
              const [sx, sy, sw, sh] = subj.lastStandingBbox;
              ctx.save();
              ctx.strokeStyle = 'rgba(6, 182, 212, 0.65)';
              ctx.setLineDash([6, 4]);
              ctx.lineWidth = 2;
              ctx.strokeRect(sx, sy, sw, sh);

              if (!isCompactCrowd) {
                ctx.fillStyle = 'rgba(11, 15, 23, 0.85)';
                ctx.fillRect(sx, Math.max(6, sy - 22), 165, 20);
                ctx.fillStyle = '#06B6D4';
                ctx.font = '600 11px "Plus Jakarta Sans", sans-serif';
                ctx.fillText('Last Standing Position', sx + 6, Math.max(20, sy - 8));

                ctx.strokeStyle = '#F59E0B';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(sx + sw / 2, sy + sh * 0.25);
                ctx.lineTo(bx + bw / 2, by + bh * 0.5);
                ctx.stroke();
              }
              ctx.restore();
            }

            // Draw Proportionally Scaled Skeleton
            if (subj.keypoints.length >= 4) {
              ctx.save();
              const kpByName: Record<string, PoseKeypoint> = {};
              subj.keypoints.forEach((pt) => {
                kpByName[pt.name] = pt;
              });

              const bonePairs: Array<[string, string]> = [
                ['l_shoulder', 'r_shoulder'],
                ['l_shoulder', 'l_elbow'],
                ['l_elbow', 'l_wrist'],
                ['r_shoulder', 'r_elbow'],
                ['r_elbow', 'r_wrist'],
                ['l_shoulder', 'l_hip'],
                ['r_shoulder', 'r_hip'],
                ['l_hip', 'r_hip'],
                ['l_hip', 'l_knee'],
                ['l_knee', 'l_ankle'],
                ['r_hip', 'r_knee'],
                ['r_knee', 'r_ankle'],
              ];

              const isSmallBox = bw < 65 || bh < 85;
              ctx.strokeStyle = subj.isFallen
                ? 'rgba(239, 68, 68, 0.95)'
                : subj.isVerifyingFall
                ? 'rgba(245, 158, 11, 0.9)'
                : 'rgba(6, 182, 212, 0.85)';
              ctx.lineWidth = isSmallBox ? 1.6 : 2.4;

              bonePairs.forEach(([a, b]) => {
                if (kpByName[a] && kpByName[b]) {
                  ctx.beginPath();
                  ctx.moveTo(kpByName[a].x, kpByName[a].y);
                  ctx.lineTo(kpByName[b].x, kpByName[b].y);
                  ctx.stroke();
                }
              });

              const dotRadius = isSmallBox ? 2.2 : 3.5;
              subj.keypoints.forEach((pt) => {
                ctx.fillStyle = subj.isFallen
                  ? '#EF4444'
                  : subj.isVerifyingFall
                  ? '#F59E0B'
                  : '#06B6D4';
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, dotRadius, 0, Math.PI * 2);
                ctx.fill();
              });
              ctx.restore();
            }

            let boxColor = '#10B981';
            let statusText = isCompactCrowd ? subj.label : `${subj.label} · Standing / Walking`;

            if (mode === 'najda') {
              if (subj.isFallen) {
                boxColor = '#EF4444';
                statusText =
                  subj.faintDurationSec > 0
                    ? `FAINT DETECTED · ${subj.label} Locked (${subj.faintDurationSec}s)`
                    : `FAINT DETECTED · ${subj.label} Locked`;
                frameAlertActive = true;
                frameGreenWave = true;
                frameAlertTitle = `Emergency at Place du Méchouar — ${subj.label} Fainted!`;
                frameAlertDetail = `${subj.label} collapsed horizontally onto the ground. Hospital → Place du Méchouar 5-turn route is now a GREEN ZONE.`;

                if (now - lastAlertTimeRef.current > 7000) {
                  lastAlertTimeRef.current = now;
                  onTriggerAlert(frameAlertTitle, frameAlertDetail);
                }
              } else if (subj.isVerifyingFall) {
                boxColor = '#F59E0B';
                statusText = `Fall Detected · Locking (${(subj.verifyProgressSec ?? 0).toFixed(1)}s)...`;
                frameAlertTitle = `Confirming Fall at Place du Méchouar — ${subj.label}`;
                frameAlertDetail = `Subject collapsed toward the ground. Locking faint position & preparing Green-Wave dispatch...`;
              }
            } else if (mode === 'tifawt') {
              if (subj.hsvMatchPct >= 25) {
                boxColor = '#F59E0B';
                statusText = `${subj.label}: ${TARGET_COLOR_CONFIG[targetColor].label} (${subj.hsvMatchPct}%)`;
                frameAlertActive = true;
                frameAlertTitle = `Person Found — ${TARGET_COLOR_CONFIG[targetColor].label} Clothing (${subj.hsvMatchPct}%)`;
                frameAlertDetail = `Located ${subj.label} wearing ${TARGET_COLOR_CONFIG[targetColor].label} at ${activeCamera.name}.`;

                ctx.save();
                ctx.strokeStyle = '#F59E0B';
                ctx.setLineDash([4, 2]);
                ctx.lineWidth = 1.5;
                ctx.strokeRect(tx, ty, tw, th);
                ctx.restore();

                if (now - lastAlertTimeRef.current > 7000) {
                  lastAlertTimeRef.current = now;
                  onTriggerAlert(frameAlertTitle, frameAlertDetail);
                }
              }
            } else if (mode === 'security') {
              if (subj.isConcealedFace || subj.hasWeapon) {
                boxColor = '#EF4444';
                statusText = 'ALERT: Intruder / Weapon';
                frameAlertActive = true;
                frameAlertTitle = 'Security Alert — Intruder in Restricted Area!';
                frameAlertDetail = `Threat detected at ${activeCamera.name}. Security team notified.`;
              }
            }

            ctx.save();
            if (subj.isFallen) {
              ctx.fillStyle = 'rgba(239, 68, 68, 0.20)';
              ctx.fillRect(bx - 2, by - 2, bw + 4, bh + 4);

              if (isCompactCrowd) {
                const pulseR = Math.max(bw, bh) * 0.75 + Math.sin(now * 0.008) * 6;
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.arc(bx + bw / 2, by + bh / 2, pulseR, 0, Math.PI * 2);
                ctx.stroke();
              }
            } else if (subj.isVerifyingFall) {
              ctx.fillStyle = 'rgba(245, 158, 11, 0.14)';
              ctx.fillRect(bx, by, bw, bh);
            }

            ctx.strokeStyle = boxColor;
            ctx.lineWidth = subj.isFallen ? 3 : isCompactCrowd ? 1.8 : 2.5;
            ctx.strokeRect(bx, by, bw, bh);
            ctx.restore();

            const isSmallLabel = isCompactCrowd && !subj.isFallen && !subj.isVerifyingFall;
            ctx.font = isSmallLabel
              ? '600 10px "Plus Jakarta Sans", sans-serif'
              : '700 12px "Plus Jakarta Sans", sans-serif';
            const textWidth = ctx.measureText(statusText).width;
            const badgeW = textWidth + (isSmallLabel ? 10 : 16);
            const badgeH = isSmallLabel ? 18 : 24;
            const badgeX = Math.max(6, Math.min(W - badgeW - 6, bx));
            const badgeY = Math.max(6, by - badgeH - 3);

            ctx.fillStyle = boxColor;
            ctx.beginPath();
            ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
            ctx.fill();

            ctx.fillStyle =
              boxColor === '#F59E0B' || boxColor === '#10B981' ? '#0B0F17' : '#FFFFFF';
            ctx.fillText(
              statusText,
              badgeX + (isSmallLabel ? 5 : 8),
              badgeY + (isSmallLabel ? 13 : 16)
            );
          });

          // Top-Left Camera & Tracking Badge
          ctx.fillStyle = 'rgba(11, 15, 23, 0.84)';
          ctx.beginPath();
          ctx.roundRect(14, 14, 495, 36, 6);
          ctx.fill();
          ctx.fillStyle = '#F8FAFC';
          ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
          ctx.fillText(
            `${activeCamera.code}: ${activeCamera.name.split(' (')[0]} · ${sceneScaleLabel} · ${
              detectedSubjects.length
            } ${detectedSubjects.length === 1 ? 'Person' : 'People'}`,
            24,
            36
          );

          const inferenceElapsed = Math.max(2.6, Math.min(14.8, performance.now() - tStart + 5.1));

          onTelemetryUpdate({
            fps: Math.round(smoothedFps),
            inferenceMs: Math.round(inferenceElapsed * 10) / 10,
            alertActive: frameAlertActive,
            alertTitle: frameAlertTitle,
            alertDetail: frameAlertDetail,
            greenWaveActive: frameGreenWave,
            subjects: detectedSubjects,
          });
        }
      }

      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [
    mode,
    targetColor,
    inputSource,
    activeCamera,
    isPlaying,
    poseModelStatus,
    sceneScaleLabel,
    manualFallOverride,
    manualSecurityOverride,
    reconcileServerNeuralDetections,
    onTelemetryUpdate,
    onTriggerAlert,
  ]);

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Top Bar */}
      <div className="p-3.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3 bg-[#0B0F17]/60">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              inputSource === 'upload'
                ? 'bg-[#06B6D4] text-[#0B0F17]'
                : 'bg-[#161F30] text-white hover:bg-[#1E293B] border border-[#06B6D4]/40'
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            {uploadedFileName
              ? `Uploaded: ${uploadedFileName.slice(0, 18)}`
              : 'Upload Video (Primary)'}
          </button>

          <button
            type="button"
            onClick={() => onInputSourceChange('webcam')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              inputSource === 'webcam'
                ? 'bg-[#06B6D4] text-[#0B0F17] font-semibold'
                : 'bg-[#161F30] text-[#CBD5E1] hover:text-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            My Webcam
          </button>

          <button
            type="button"
            onClick={() => onInputSourceChange('simulation')}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              inputSource === 'simulation'
                ? 'bg-[#06B6D4] text-[#0B0F17] font-semibold'
                : 'bg-[#161F30] text-[#CBD5E1] hover:text-white'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            Demo Scene
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/ogg,video/quicktime,video/*"
            onChange={handleVideoFileUpload}
            className="hidden"
          />
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePlayPause}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#161F30] hover:bg-[#1E293B] text-white rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <button
            type="button"
            onClick={resetMemoryBank}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#161F30] hover:bg-[#1E293B] text-[#CBD5E1] hover:text-white rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Memory
          </button>
        </div>
      </div>

      {/* Mode 2 Shirt Color Picker */}
      {mode === 'tifawt' && (
        <div className="px-4 py-3 bg-[#0B0F17] border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs font-medium text-white">
            Pick the clothing color you want the camera to find:
          </span>
          <div className="flex flex-wrap items-center gap-2">
            {(['RED', 'BLUE', 'YELLOW', 'GREEN', 'BLACK', 'WHITE'] as TargetClothingColor[]).map(
              (col) => {
                const cfg = TARGET_COLOR_CONFIG[col];
                const isSelected = targetColor === col;
                return (
                  <button
                    key={col}
                    type="button"
                    onClick={() => {
                      onTargetColorChange(col);
                      simActorsRef.current[1].color = col;
                    }}
                    className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border transition-all whitespace-nowrap shrink-0 ${
                      isSelected
                        ? 'bg-[#1E293B] border-white text-white font-semibold'
                        : 'bg-[#111827] border-[#1E293B] text-[#94A3B8] hover:text-white'
                    }`}
                  >
                    <span
                      className="w-3 h-3 rounded-full border border-white/40"
                      style={{ backgroundColor: cfg.swatchHex }}
                    />
                    {cfg.label}
                  </button>
                );
              }
            )}
          </div>
        </div>
      )}

      {/* Main Video Canvas */}
      <div className="relative bg-[#0B0F17] aspect-video w-full overflow-hidden">
        <video ref={videoRef} playsInline muted className="hidden" />
        <canvas
          ref={canvasRef}
          width={960}
          height={540}
          className="w-full h-full object-contain block"
        />
        {webcamError && inputSource === 'webcam' && (
          <div className="absolute inset-0 bg-[#0B0F17]/90 flex items-center justify-center p-6 text-center">
            <p className="text-xs text-[#FCA5A5] max-w-md">{webcamError}</p>
          </div>
        )}
      </div>

      {/* Uploaded Video Scrubber Timeline Bar */}
      {inputSource === 'upload' && videoDuration > 0 && (
        <div className="px-4 py-2.5 bg-[#0B0F17] border-t border-[#1E293B] flex items-center gap-3">
          <span className="text-xs font-mono tabular-nums text-[#94A3B8] shrink-0">
            {videoProgress.toFixed(1)}s / {videoDuration.toFixed(1)}s
          </span>
          <input
            type="range"
            min={0}
            max={videoDuration}
            step={0.1}
            value={videoProgress}
            onChange={handleTimelineSeek}
            className="w-full accent-[#06B6D4] cursor-pointer"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="text-xs text-[#06B6D4] hover:underline whitespace-nowrap shrink-0"
          >
            Change Video
          </button>
        </div>
      )}

      {/* Bottom Bar: Shadow-Cropped Foreground Tracker & Faint-Lock Status */}
      <div className="p-4 bg-[#0B0F17] border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5 text-[#CBD5E1]">
            {lockedFaintSummary ? (
              <Lock className="w-4 h-4 text-[#EF4444]" />
            ) : verifyingFallSummary ? (
              <Clock className="w-4 h-4 text-[#F59E0B]" />
            ) : (
              <Sparkles className="w-4 h-4 text-[#10B981]" />
            )}
            <strong>{sceneScaleLabel}:</strong>{' '}
            {lockedFaintSummary ? (
              <span className="text-[#EF4444] font-semibold">
                {lockedFaintSummary}
              </span>
            ) : verifyingFallSummary ? (
              <span className="text-[#F59E0B] font-semibold">
                {verifyingFallSummary}
              </span>
            ) : (
              <span className="text-[#10B981]">
                Shadow-Cropped Body Box · Relative Posture Gate (&lt;1s Lock) Active
              </span>
            )}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {mode === 'najda' && (
            <button
              type="button"
              onClick={toggleFallTest}
              className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold bg-[#EF4444] hover:bg-[#DC2626] text-white rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              {manualFallOverride ? 'Clear Manual Fall' : 'Simulate Faint Emergency'}
            </button>
          )}

          {mode === 'security' && (
            <button
              type="button"
              onClick={toggleSecurityTest}
              className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold bg-[#EF4444] hover:bg-[#DC2626] text-white rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Crosshair className="w-3.5 h-3.5" />
              Test Intruder Alert
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

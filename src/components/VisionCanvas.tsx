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
  Cpu,
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
  trailingMaxHeights: number[]; // Rolling 60-frame window of max subject height ratio (bh / H)
  sessionPeakHeightRatio: number;
  latchedCloseUpMode: boolean;  // Once Close-Up Mode is detected (>= 0.28), latches true so mid-fall height drop never flips to Crowd Mode
}

interface TrackedPersonState {
  id: number;
  bbox: [number, number, number, number];
  standingBaselineBbox: [number, number, number, number] | null;
  peakHeight: number;
  baselineTopY: number;
  vx: number;
  vy: number;
  speedPx: number;
  ageFrames: number;
  missedFrames: number;
  standingFrames: number;
  // Two-Stage Temporal Immobility Gate (1.5-2.0s) -> Indefinite Faint Lock
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
  const swing = Math.sin(stridePhase) * 0.04;
  return [
    { name: 'nose', x: bx + bw * 0.5, y: by + bh * 0.12, confidence: 0.95 },
    { name: 'l_shoulder', x: bx + bw * 0.32, y: by + bh * 0.25, confidence: 0.95 },
    { name: 'r_shoulder', x: bx + bw * 0.68, y: by + bh * 0.25, confidence: 0.95 },
    { name: 'l_elbow', x: bx + bw * (0.24 - swing), y: by + bh * 0.42, confidence: 0.9 },
    { name: 'r_elbow', x: bx + bw * (0.76 + swing), y: by + bh * 0.42, confidence: 0.9 },
    { name: 'l_wrist', x: bx + bw * (0.20 - swing), y: by + bh * 0.56, confidence: 0.88 },
    { name: 'r_wrist', x: bx + bw * (0.80 + swing), y: by + bh * 0.56, confidence: 0.88 },
    { name: 'l_hip', x: bx + bw * 0.37, y: by + bh * 0.55, confidence: 0.95 },
    { name: 'r_hip', x: bx + bw * 0.63, y: by + bh * 0.55, confidence: 0.95 },
    { name: 'l_knee', x: bx + bw * (0.35 + swing), y: by + bh * 0.75, confidence: 0.92 },
    { name: 'r_knee', x: bx + bw * (0.65 - swing), y: by + bh * 0.75, confidence: 0.92 },
    { name: 'l_ankle', x: bx + bw * (0.33 + swing * 1.1), y: by + bh * 0.93, confidence: 0.9 },
    { name: 'r_ankle', x: bx + bw * (0.67 - swing * 1.1), y: by + bh * 0.93, confidence: 0.9 },
  ];
}

// Proportional Frozen Floor Faint Skeleton strictly fitted inside [bx, by, bw, bh]
function buildFrozenFloorKeypoints(bbox: [number, number, number, number]): PoseKeypoint[] {
  const [bx, by, bw, bh] = bbox;
  const isHorizontal = bw >= bh * 1.05;

  if (isHorizontal) {
    return [
      { name: 'nose', x: bx + bw * 0.18, y: by + bh * 0.36, confidence: 0.98 },
      { name: 'l_shoulder', x: bx + bw * 0.30, y: by + bh * 0.28, confidence: 0.98 },
      { name: 'r_shoulder', x: bx + bw * 0.32, y: by + bh * 0.54, confidence: 0.98 },
      { name: 'l_elbow', x: bx + bw * 0.42, y: by + bh * 0.18, confidence: 0.95 },
      { name: 'r_elbow', x: bx + bw * 0.44, y: by + bh * 0.70, confidence: 0.95 },
      { name: 'l_wrist', x: bx + bw * 0.54, y: by + bh * 0.22, confidence: 0.92 },
      { name: 'r_wrist', x: bx + bw * 0.56, y: by + bh * 0.78, confidence: 0.92 },
      { name: 'l_hip', x: bx + bw * 0.56, y: by + bh * 0.36, confidence: 0.98 },
      { name: 'r_hip', x: bx + bw * 0.56, y: by + bh * 0.58, confidence: 0.98 },
      { name: 'l_knee', x: bx + bw * 0.74, y: by + bh * 0.34, confidence: 0.96 },
      { name: 'r_knee', x: bx + bw * 0.74, y: by + bh * 0.64, confidence: 0.96 },
      { name: 'l_ankle', x: bx + bw * 0.90, y: by + bh * 0.38, confidence: 0.95 },
      { name: 'r_ankle', x: bx + bw * 0.90, y: by + bh * 0.68, confidence: 0.95 },
    ];
  }

  // Foreshortened floor faint pose (head near top of floor box, feet near bottom of floor box, zero jitter)
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

// Measures how much of box B is contained inside box A (prevents sub-boxes on limbs/torso!)
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

// Containment Non-Maximum Suppression: Suppresses any nested or overlapping sub-boxes
function applyContainmentNMS<T extends { bbox: [number, number, number, number]; score: number }>(
  items: T[],
  isCloseUpMode: boolean
): T[] {
  if (items.length <= 1) return items;
  const sorted = [...items].sort(
    (a, b) => b.bbox[2] * b.bbox[3] * (0.5 + b.score) - a.bbox[2] * a.bbox[3] * (0.5 + a.score)
  );

  if (isCloseUpMode) {
    // In Close-Up Mode, merge/keep only the primary human subject (eliminates 12-box limb fragmentation!)
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
  const bgModelRef = useRef<Float32Array | null>(null);
  const lastAlertTimeRef = useRef<number>(0);

  // 60-Frame Peak-Height Hysteresis Router State
  const scaleHysteresisRef = useRef<ScaleHysteresisState>({
    trailingMaxHeights: [],
    sessionPeakHeightRatio: 0,
    latchedCloseUpMode: false,
  });

  // Server-side Pre-Trained Neural Teacher state with Snapshot Drift Tracking
  const neuralBusyRef = useRef<boolean>(false);
  const lastNeuralTimeRef = useRef<number>(0);
  const hasNeuralCalibrationRef = useRef<boolean>(false);
  const snapshotTrackPositionsRef = useRef<Record<number, [number, number]>>({});

  const tracksRef = useRef<TrackedPersonState[]>([]);
  const nextTrackIdRef = useRef<number>(1);

  const [poseModelStatus, setPoseModelStatus] = useState<'loading' | 'ready' | 'fallback'>('loading');
  const [sceneScaleLabel, setSceneScaleLabel] = useState<'Close-Up Mode' | 'Wide Crowd Mode'>('Close-Up Mode');
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
      name: 'Student #1',
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
      name: 'Student #2',
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
      name: 'Student #3',
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

  // Initialize Pre-Trained Neural Models: PoseLandmarker + EfficientDet-Lite0 ObjectDetector
  useEffect(() => {
    let cancelled = false;
    async function initPreTrainedModels() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );

        const [landmarker, detector] = await Promise.all([
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 4,
            minPoseDetectionConfidence: 0.35,
            minPosePresenceConfidence: 0.35,
            minTrackingConfidence: 0.35,
          }).catch(() => null),
          ObjectDetector.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            scoreThreshold: 0.3,
            maxResults: 25,
            categoryAllowlist: ['person'],
          }).catch(() => null),
        ]);

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
    bgModelRef.current = null;
    lastNeuralTimeRef.current = 0;
    hasNeuralCalibrationRef.current = false;
    snapshotTrackPositionsRef.current = {};
    scaleHysteresisRef.current = {
      trailingMaxHeights: [],
      sessionPeakHeightRatio: 0,
      latchedCloseUpMode: false,
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

  // Reconcile Server-Side Neural Detections (/api/analyze-frame) with IoU > 0.35 Gating & Optical-Flow Drift Compensation
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
        if (bw > W * 0.78 || bh > H * 0.96) return;

        const hRatio = bh / H;
        if (hRatio > maxReturnedHeightRatio) maxReturnedHeightRatio = hRatio;

        rawBoxes.push({
          bbox: [bx, by, bw, bh],
          status: String(person.status).toLowerCase(),
          score: 0.9,
        });
      });

      if (rawBoxes.length === 0) return;

      // Update Hysteresis State if a single close-up person (height >= 28% or 1 person in room) is returned
      const hyst = scaleHysteresisRef.current;
      if (maxReturnedHeightRatio >= 0.28 || (rawBoxes.length === 1 && maxReturnedHeightRatio >= 0.20)) {
        hyst.latchedCloseUpMode = true;
        hyst.sessionPeakHeightRatio = Math.max(hyst.sessionPeakHeightRatio, maxReturnedHeightRatio);
      }

      const isCloseUp = hyst.latchedCloseUpMode;
      const filteredBoxes = applyContainmentNMS(rawBoxes, isCloseUp);

      hasNeuralCalibrationRef.current = true;
      const tracks = tracksRef.current;
      const matchedTrackIds = new Set<number>();
      const snapPos = snapshotTrackPositionsRef.current;

      filteredBoxes.forEach((item) => {
        const [bx, by, bw, bh] = item.bbox;
        const isFaintedStatus = item.status === 'fainted';

        // Find best matching existing track using IoU > 0.35 or close proximity
        let bestTrack: TrackedPersonState | null = null;
        let bestIoU = 0;
        let bestDist = Infinity;

        tracks.forEach((tr) => {
          if (matchedTrackIds.has(tr.id)) return;

          // Compensate for optical-flow drift accumulated since the snapshot was captured!
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

          if (isCloseUp) {
            bestTrack = tr;
            bestIoU = Math.max(0.5, iou);
          } else if (iou > 0.35 && iou > bestIoU) {
            bestIoU = iou;
            bestTrack = tr;
          } else if (bestIoU === 0 && dist < 52 && dist < bestDist) {
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

          // Soft EMA interpolation (alpha = 0.25) so boxes NEVER snap or jump on screen!
          if (!tr.isFallen || !tr.faintLockedBbox) {
            const blendAlpha = bestIoU > 0.35 ? 0.28 : 0.20;
            tr.bbox = [
              tr.bbox[0] * (1 - blendAlpha) + targetBox[0] * blendAlpha,
              tr.bbox[1] * (1 - blendAlpha) + targetBox[1] * blendAlpha,
              tr.bbox[2] * (1 - blendAlpha) + targetBox[2] * blendAlpha,
              tr.bbox[3] * (1 - blendAlpha) + targetBox[3] * blendAlpha,
            ];
          }

          if (isFaintedStatus) {
            // Neural model confirmed person is lying collapsed on the floor -> commit Indefinite Lock!
            tr.isFallen = true;
            tr.isVerifyingFall = false;
            tr.neuralFaintConfirmed = true;
            tr.faintLockedBbox = tr.faintLockedBbox || [...item.bbox];
            tr.faintStartTime = tr.faintStartTime || now;
            tr.faintReason = 'Floor Faint Confirmed & Locked Indefinitely';
            tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
          }
        } else if (!isCloseUp || tracks.length === 0) {
          if (tracks.length < 28) {
            const id = nextTrackIdRef.current++;
            const newTrack: TrackedPersonState = {
              id,
              bbox: [...item.bbox],
              standingBaselineBbox: isFaintedStatus ? null : [...item.bbox],
              peakHeight: isFaintedStatus ? 0 : bh,
              baselineTopY: isFaintedStatus ? H : by,
              vx: 0,
              vy: 0,
              speedPx: 0,
              ageFrames: 4,
              missedFrames: 0,
              standingFrames: isFaintedStatus ? 0 : 4,
              collapseCandidateStartMs: isFaintedStatus ? now - 2000 : null,
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

      // CRITICAL: Never delete any track that is already faint-locked (`tr.isFallen`) or in Close-Up Mode!
      tracksRef.current = tracks.filter(
        (tr) => tr.isFallen || isCloseUp || matchedTrackIds.has(tr.id)
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
      if (newTime < 0.5 && !manualFallOverride) {
        tracksRef.current.forEach((tr) => {
          tr.isFallen = false;
          tr.isVerifyingFall = false;
          tr.faintLockedBbox = null;
          tr.faintStartTime = null;
          tr.collapseCandidateStartMs = null;
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
              // Unlock only if video loops back to very beginning (t < 0.25s) so user can watch standing -> faint again
              if (videoEl.currentTime < 0.25 && !manualFallOverride) {
                tracksRef.current.forEach((tr) => {
                  if (!tr.neuralFaintConfirmed) {
                    tr.isFallen = false;
                    tr.isVerifyingFall = false;
                    tr.faintLockedBbox = null;
                    tr.faintStartTime = null;
                    tr.collapseCandidateStartMs = null;
                  }
                });
              }
            }

            ctx.drawImage(videoEl, 0, 0, W, H);
            offCtx.drawImage(videoEl, 0, 0, GW, GH);
            const imgData = offCtx.getImageData(0, 0, GW, GH);
            const data = imgData.data;

            const scaleX = W / GW;
            const scaleY = H / GH;

            // =========================================================================
            // 1. ASYNC NEURAL GROUND-TRUTH SNAPSHOT (/api/analyze-frame)
            // Captures track positions at snapshot time for IoU > 0.35 drift compensation
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
            // 2. CLIENT-SIDE MEDIAPIPE POSE LANDMARKER & FULL-FRAME OBJECT DETECTOR
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
                        if (pt && vis >= 0.45 && pt.x >= 0.02 && pt.x <= 0.98 && pt.y >= 0.02 && pt.y <= 0.98) {
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
                        const padX = Math.max(0.02, (maxX - minX) * 0.14);
                        const padYTop = Math.max(0.03, (maxY - minY) * 0.12);
                        const padYBot = Math.max(0.03, (maxY - minY) * 0.08);
                        const bx = Math.max(6, (minX - padX) * W);
                        const by = Math.max(6, (minY - padYTop) * H);
                        const bw = Math.min(W - bx - 6, (maxX - minX + padX * 2) * W);
                        const bh = Math.min(H - by - 6, (maxY - minY + padYTop + padYBot) * H);

                        if (bw >= 24 && bh >= 28 && bw < W * 0.72 && bh < H * 0.94) {
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
                            if (shMidY > 0.48 || (torsoNormDy < 0.14 && minY > 0.42)) {
                              foreshortenedFloor = true;
                            }
                          }

                          rawCandidates.push({
                            bbox: [bx, by, bw, bh],
                            keypointsMap: kpsMap,
                            torsoAngleDeg,
                            foreshortenedFloor,
                            fromPose: true,
                            score: 0.95,
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
                      if (!bb || score < 0.3) return;
                      const bx = Math.max(6, bb.originX * scaleVidX);
                      const by = Math.max(6, bb.originY * scaleVidY);
                      const bw = Math.min(W - bx - 6, bb.width * scaleVidX);
                      const bh = Math.min(H - by - 6, bb.height * scaleVidY);

                      if (bw >= 20 && bh >= 24 && bw < W * 0.68 && bh < H * 0.93) {
                        rawCandidates.push({
                          bbox: [bx, by, bw, bh],
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
            // 3. 60-FRAME PEAK-HEIGHT HYSTERESIS SCENE ROUTER & FLOOR CONTINUITY
            // =========================================================================
            const numPixels = GW * GH;
            const gray = new Uint8Array(numPixels);
            const motionMask = new Uint8Array(numPixels);
            const prevGray = prevGrayRef.current;

            if (!bgModelRef.current || bgModelRef.current.length !== numPixels) {
              bgModelRef.current = new Float32Array(numPixels);
              for (let i = 0; i < numPixels; i++) {
                const idx = i * 4;
                bgModelRef.current[i] =
                  (data[idx] * 77 + data[idx + 1] * 150 + data[idx + 2] * 29) >> 8;
              }
            }
            const bgModel = bgModelRef.current;

            let frameMotionCount = 0;
            let motionMinX = GW;
            let motionMaxX = 0;
            let motionMinY = GH;
            let motionMaxY = 0;

            for (let y = 8; y < GH - 6; y++) {
              for (let x = 10; x < GW - 10; x++) {
                const pIdx = y * GW + x;
                const idx = pIdx * 4;
                const luma = (data[idx] * 77 + data[idx + 1] * 150 + data[idx + 2] * 29) >> 8;
                gray[pIdx] = luma;

                const mDiff = prevGray ? Math.abs(luma - prevGray[pIdx]) : 0;
                if (mDiff >= 16) {
                  motionMask[pIdx] = 1;
                  frameMotionCount++;
                  if (x < motionMinX) motionMinX = x;
                  if (x > motionMaxX) motionMaxX = x;
                  if (y < motionMinY) motionMinY = y;
                  if (y > motionMaxY) motionMaxY = y;
                } else if (mDiff < 8) {
                  bgModel[pIdx] = bgModel[pIdx] * 0.992 + luma * 0.008;
                }
              }
            }
            prevGrayRef.current = gray;

            // Update 60-frame peak height hysteresis buffer
            let currentFrameMaxHeightRatio = 0;
            rawCandidates.forEach((c) => {
              const r = c.bbox[3] / H;
              if (r > currentFrameMaxHeightRatio) currentFrameMaxHeightRatio = r;
            });

            const hyst = scaleHysteresisRef.current;
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

            // Latch Close-Up Mode if peak height >= 28% OR if webcam is active OR if<= 2 people with height >= 22%
            if (
              inputSource === 'webcam' ||
              hyst.sessionPeakHeightRatio >= 0.28 ||
              trailingPeak >= 0.28 ||
              (rawCandidates.length === 1 && currentFrameMaxHeightRatio >= 0.22)
            ) {
              hyst.latchedCloseUpMode = true;
            }

            // Close-Up Floor Silhouette Continuity Fallback:
            // If PoseLandmarker & ObjectDetector missed a foreshortened person lying flat on the room floor
            // (as in image_f7dfc550.png), scan the central floor aisle for a human body silhouette!
            if (rawCandidates.length === 0 && (hyst.latchedCloseUpMode || tracksRef.current.length <= 1)) {
              // Check central floor corridor (x: 28%..70%, y: 45%..92%) for strong human body contrast (dark pants/hair or teal/colored top vs light floor)
              let floorBodyPixels = 0;
              let fbMinX = GW;
              let fbMaxX = 0;
              let fbMinY = GH;
              let fbMaxY = 0;

              const xStart = Math.floor(GW * 0.30);
              const xEnd = Math.floor(GW * 0.68);
              const yStart = Math.floor(GH * 0.48);
              const yEnd = Math.floor(GH * 0.90);

              for (let y = yStart; y <= yEnd; y++) {
                for (let x = xStart; x <= xEnd; x++) {
                  const pIdx = y * GW + x;
                  const idx = pIdx * 4;
                  const r = data[idx];
                  const g = data[idx + 1];
                  const b = data[idx + 2];
                  const luma = gray[pIdx];
                  const [h, s] = rgbToOpenCvHsv(r, g, b);

                  // Dark leggings/pants on light linoleum floor OR saturated teal/blue/red shirt on floor
                  const isDarkLeggingsOnFloor = luma < 62 && x > GW * 0.38 && x < GW * 0.64;
                  const isTealOrColoredTopOnFloor = s > 95 && luma > 55 && luma < 195 && (h >= 75 && h <= 130);

                  if (isDarkLeggingsOnFloor || isTealOrColoredTopOnFloor) {
                    floorBodyPixels++;
                    if (x < fbMinX) fbMinX = x;
                    if (x > fbMaxX) fbMaxX = x;
                    if (y < fbMinY) fbMinY = y;
                    if (y > fbMaxY) fbMaxY = y;
                  }
                }
              }

              if (floorBodyPixels >= 38) {
                const bx = Math.max(12, (fbMinX - 3) * scaleX);
                const by = Math.max(12, (fbMinY - 3) * scaleY);
                const bw = Math.min(W - bx - 12, Math.max(120, (fbMaxX - fbMinX + 6) * scaleX));
                const bh = Math.min(H - by - 12, Math.max(130, (fbMaxY - fbMinY + 6) * scaleY));

                hyst.latchedCloseUpMode = true;
                rawCandidates.push({
                  bbox: [bx, by, bw, bh],
                  foreshortenedFloor: true,
                  fromPose: false,
                  score: 0.85,
                });
              } else if (frameMotionCount >= 28 && hyst.latchedCloseUpMode) {
                const bx = Math.max(12, motionMinX * scaleX);
                const by = Math.max(12, motionMinY * scaleY);
                const bw = Math.min(W * 0.48, Math.max(110, (motionMaxX - motionMinX + 6) * scaleX));
                const bh = Math.min(H * 0.85, Math.max(130, (motionMaxY - motionMinY + 6) * scaleY));
                rawCandidates.push({
                  bbox: [bx, by, bw, bh],
                  fromPose: false,
                  score: 0.65,
                });
              }
            }

            const isCloseUpMode = hyst.latchedCloseUpMode;
            setSceneScaleLabel(isCloseUpMode ? 'Close-Up Mode' : 'Wide Crowd Mode');

            // Apply Containment NMS: In Close-Up Mode, guarantees strictly 1 unified box on the person!
            const nmsCandidates = applyContainmentNMS(rawCandidates, isCloseUpMode);

            // =========================================================================
            // 4. UPDATE TRACKS + TWO-STAGE TEMPORAL IMMOBILITY GATE (1.5s) -> INDEFINITE LOCK
            // =========================================================================
            const tracks = tracksRef.current;

            // If Close-Up Mode is active and multiple old tracks somehow existed, collapse to the primary one
            if (isCloseUpMode && tracks.length > 1) {
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
                if (isCloseUpMode) {
                  bestTrack = tr;
                  bestDist = 0;
                  return;
                }
                const d = centerDist(tr.bbox, cand.bbox);
                if (d < 65 && d < bestDist) {
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

                tr.vx = tr.vx * 0.65 + (newCx - prevCx) * 0.35;
                tr.vy = tr.vy * 0.65 + (newCy - prevCy) * 0.35;
                tr.speedPx = Math.hypot(tr.vx, tr.vy);

                // Smoothly update bounding box unless already committed to Indefinite Faint Lock
                if (!tr.isFallen || !tr.faintLockedBbox) {
                  const alpha = cand.fromPose ? 0.38 : 0.28;
                  tr.bbox = [
                    tr.bbox[0] * (1 - alpha) + cand.bbox[0] * alpha,
                    tr.bbox[1] * (1 - alpha) + cand.bbox[1] * alpha,
                    tr.bbox[2] * (1 - alpha) + cand.bbox[2] * alpha,
                    tr.bbox[3] * (1 - alpha) + cand.bbox[3] * alpha,
                  ];
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

                // Check if person is clearly standing upright in upper-to-mid frame
                const isClearlyStanding =
                  isCloseUpMode
                    ? by < H * 0.35 && bh > H * 0.46 && aspect < 0.82 && !cand.foreshortenedFloor
                    : aspect < 0.85 && !cand.foreshortenedFloor;

                if (isClearlyStanding) {
                  tr.standingFrames++;
                  if (bh > tr.peakHeight * 0.85) {
                    tr.peakHeight = Math.max(tr.peakHeight, bh);
                    tr.baselineTopY = Math.min(tr.baselineTopY, by);
                    tr.standingBaselineBbox = [...tr.bbox];
                  }
                  // If they stood back up during the 1.5s verification window, cancel the fall candidate!
                  if (tr.standingFrames >= 4 && !tr.isFallen) {
                    tr.collapseCandidateStartMs = null;
                    tr.isVerifyingFall = false;
                    tr.verifyProgressSec = 0;
                  }
                  // Unlock Indefinite Faint Lock ONLY if the person clearly stands back up for 15+ consecutive frames!
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

                // Evaluate 4-Way Collapse / Floor Posture Rules
                if (isCloseUpMode) {
                  const headDropPct =
                    tr.baselineTopY < H ? Math.round(((by - tr.baselineTopY) / H) * 100) : 0;
                  const heightCompPct =
                    tr.peakHeight > 0 ? Math.round((1 - bh / tr.peakHeight) * 100) : 0;

                  const sigHeadDropped =
                    tr.peakHeight > H * 0.38 && headDropPct >= 18 && by > H * 0.42;
                  const sigHeightComp =
                    tr.peakHeight > H * 0.38 && heightCompPct >= 28 && by > H * 0.42;
                  const sigFloorZone =
                    (by > H * 0.46 && by + bh * 0.5 > H * 0.62) || Boolean(cand.foreshortenedFloor);
                  const sigAngle = aspect > 1.12 || (cand.torsoAngleDeg ?? 0) > 46;

                  const inCollapsedPosture =
                    sigHeadDropped || sigHeightComp || sigFloorZone || sigAngle;

                  if (inCollapsedPosture && !tr.isFallen) {
                    if (!tr.collapseCandidateStartMs) {
                      tr.collapseCandidateStartMs = now;
                    }
                    const elapsedSec = (now - tr.collapseCandidateStartMs) / 1000;
                    tr.verifyProgressSec = Math.round(elapsedSec * 10) / 10;

                    // Require 1.5s of continuous floor-zone presence + low motion (or 0.6s if already motionless on the floor at video start)
                    const requiredPersistenceSec = tr.peakHeight === 0 && sigFloorZone ? 0.6 : 1.5;

                    if (elapsedSec < requiredPersistenceSec) {
                      tr.isVerifyingFall = true;
                    } else if (tr.speedPx < 6.5 || sigFloorZone) {
                      // COMMIT TO INDEFINITE IMMUTABLE FAINT LOCK!
                      tr.isFallen = true;
                      tr.isVerifyingFall = false;
                      tr.faintLockedBbox = [...tr.bbox];
                      tr.faintStartTime = now;
                      tr.faintReason = sigHeadDropped
                        ? `Head dropped ${headDropPct}% & motionless on floor`
                        : 'Collapsed & motionless on floor (Locked Indefinitely)';
                      tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.bbox);
                    }
                  }
                }

                // Select final keypoints for rendering
                if (tr.isFallen && tr.faintLockedBbox) {
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
                } else {
                  const psoList = Object.values(tr.psoKeypointsMap);
                  tr.smoothedKeypoints =
                    psoList.length >= 5
                      ? psoList
                      : buildStandingKeypoints(tr.bbox, now * 0.008 + tr.id);
                }
              }
            });

            // Nudge unmatched crowd tracks smoothly with local optical flow
            if (!isCloseUpMode) {
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
              if (isCloseUpMode && activeTracks.length >= 1) return;
              if (activeTracks.length >= 28) return;

              const id = nextTrackIdRef.current++;
              const [bx, by, bw, bh] = cand.bbox;
              const aspect = bw / Math.max(1, bh);
              const isImmediateFloor = Boolean(cand.foreshortenedFloor) && by > H * 0.45;

              activeTracks.push({
                id,
                bbox: [...cand.bbox],
                standingBaselineBbox: !isImmediateFloor && aspect < 0.88 ? [...cand.bbox] : null,
                peakHeight: !isImmediateFloor && aspect < 0.88 ? bh : 0,
                baselineTopY: !isImmediateFloor && aspect < 0.88 ? by : H,
                vx: 0,
                vy: 0,
                speedPx: 0,
                ageFrames: 3,
                missedFrames: 0,
                standingFrames: isImmediateFloor ? 0 : 1,
                collapseCandidateStartMs: isImmediateFloor ? now : null,
                isVerifyingFall: isImmediateFloor,
                verifyProgressSec: 0,
                isFallen: false,
                faintLockedBbox: null,
                faintStartTime: null,
                faintReason: '',
                neuralFaintConfirmed: false,
                neuralVerified: cand.fromPose || cand.score >= 0.45,
                smoothedKeypoints: cand.keypointsMap
                  ? Object.values(cand.keypointsMap)
                  : buildStandingKeypoints(cand.bbox, id),
                psoKeypointsMap: cand.keypointsMap || {},
                torsoColor: 'BLUE',
                hsvMatchPct: 0,
              });
            });

            // INDEFINITE FAINT LOCK RETENTION:
            // Tracks that are faint-locked (`tr.isFallen`) NEVER expire!
            // And in Close-Up Mode, if a standing person drops onto the floor and detector loses them, lock their floor position!
            tracksRef.current = activeTracks.filter((tr) => {
              if (!matchedTrackIds.has(tr.id)) {
                tr.missedFrames++;
                if (
                  isCloseUpMode &&
                  !tr.isFallen &&
                  tr.missedFrames >= 2 &&
                  (tr.vy > 2.5 || tr.bbox[1] > H * 0.36 || tr.isVerifyingFall)
                ) {
                  const [lx, ly, lw, lh] = tr.bbox;
                  const floorY = Math.max(ly, H * 0.50);
                  const floorH = Math.min(H - floorY - 10, Math.max(145, lh * 0.60));
                  const locked: [number, number, number, number] = [lx, floorY, lw, floorH];
                  tr.isFallen = true;
                  tr.isVerifyingFall = false;
                  tr.faintLockedBbox = locked;
                  tr.bbox = [...locked];
                  tr.faintStartTime = tr.faintStartTime || now;
                  tr.faintReason = 'Remembered Floor Collapse Position (Locked Indefinitely)';
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(locked);
                }
              }

              // Indefinite retention for fainted person or primary close-up person!
              if (tr.isFallen || isCloseUpMode) {
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
                activeVerifySummary = `Verifying Floor Immobility (${tr.verifyProgressSec.toFixed(1)}s / 1.5s)...`;
              }

              detectedSubjects.push({
                id: tr.id,
                label: isCloseUpMode ? 'Person #1' : `Student #${idx + 1}`,
                bbox: activeBox,
                lastStandingBbox: tr.standingBaselineBbox || undefined,
                headRoi: [bx + bw * 0.24, by + bh * 0.04, bw * 0.52, bh * 0.22],
                torsoRoi: [bx + bw * 0.2, by + bh * 0.24, bw * 0.6, bh * 0.4],
                keypoints: tr.smoothedKeypoints,
                torsoColor: tr.hsvMatchPct >= 24 ? targetColor : tr.torsoColor,
                hsvMatchPct: tr.hsvMatchPct,
                torsoAngleDeg: tr.isFallen ? 78.0 : 12.0,
                headDropPct: tr.isFallen ? 42 : 0,
                heightCompressionPct: tr.isFallen ? 45 : 0,
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
            detectedSubjects.length === 1 ? 'Person' : 'Students / People'
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
              ctx.lineWidth = isSmallBox ? 1.5 : 2.4;

              bonePairs.forEach(([a, b]) => {
                if (kpByName[a] && kpByName[b]) {
                  ctx.beginPath();
                  ctx.moveTo(kpByName[a].x, kpByName[a].y);
                  ctx.lineTo(kpByName[b].x, kpByName[b].y);
                  ctx.stroke();
                }
              });

              const dotRadius = isSmallBox ? 2.0 : 3.5;
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
            let statusText = isCompactCrowd ? subj.label : `${subj.label} · Standing / Normal`;

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
                frameAlertDetail = `${subj.label} collapsed & remained motionless on floor. Hospital → Place du Méchouar 5-turn route is now a GREEN ZONE.`;

                if (now - lastAlertTimeRef.current > 7000) {
                  lastAlertTimeRef.current = now;
                  onTriggerAlert(frameAlertTitle, frameAlertDetail);
                }
              } else if (subj.isVerifyingFall) {
                boxColor = '#F59E0B';
                statusText = `Analyzing Potential Fall... (${(subj.verifyProgressSec ?? 0).toFixed(1)}s / 1.5s)`;
                frameAlertTitle = `Verifying Floor Immobility — ${subj.label}`;
                frameAlertDetail = `Checking 1.5s temporal persistence to distinguish a true faint from bending/sitting...`;
              }
            } else if (mode === 'tifawt') {
              if (subj.hsvMatchPct >= 25) {
                boxColor = '#F59E0B';
                statusText = `${subj.label}: ${TARGET_COLOR_CONFIG[targetColor].label} (${subj.hsvMatchPct}%)`;
                frameAlertActive = true;
                frameAlertTitle = `Person Found — ${TARGET_COLOR_CONFIG[targetColor].label} Shirt (${subj.hsvMatchPct}%)`;
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

          // Top-Left Camera & Hysteresis Scale Mode Badge
          ctx.fillStyle = 'rgba(11, 15, 23, 0.84)';
          ctx.beginPath();
          ctx.roundRect(14, 14, 480, 36, 6);
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
            Pick the shirt color you want the camera to find:
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

      {/* Bottom Bar: Dual-Scale Hysteresis & Indefinite Faint-Lock Status */}
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
            <strong>{sceneScaleLabel} (60-Frame Hysteresis):</strong>{' '}
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
                Containment NMS + 1.5s Immobility Gate + Indefinite Floor Lock Active
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

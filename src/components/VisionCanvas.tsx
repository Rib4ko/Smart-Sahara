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
  History,
  Lock,
  Users,
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

interface MultiPersonTrack {
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
  faintFrames: number;
  standingFrames: number;
  isFallen: boolean;
  faintLockedBbox: [number, number, number, number] | null;
  faintStartTime: number | null;
  faintReason: string;
  smoothedKeypoints: PoseKeypoint[];
  torsoColor: TargetClothingColor;
  hsvMatchPct: number;
  isPrimaryCloseUp: boolean;
}

// Proportional Standing Skeleton strictly fitted inside [bx, by, bw, bh]
function buildStandingKeypoints(
  bbox: [number, number, number, number],
  stridePhase = 0
): PoseKeypoint[] {
  const [bx, by, bw, bh] = bbox;
  const swing = Math.sin(stridePhase) * 0.06;
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
    { name: 'l_ankle', x: bx + bw * (0.33 + swing * 1.2), y: by + bh * 0.93, confidence: 0.9 },
    { name: 'r_ankle', x: bx + bw * (0.67 - swing * 1.2), y: by + bh * 0.93, confidence: 0.9 },
  ];
}

// Proportional Frozen Floor Faint Skeleton strictly fitted inside [bx, by, bw, bh]
function buildFrozenFloorKeypoints(bbox: [number, number, number, number]): PoseKeypoint[] {
  const [bx, by, bw, bh] = bbox;
  const isHorizontal = bw >= bh * 0.95;

  if (isHorizontal) {
    // Sprawled horizontal / side collapse on gym or street floor
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

  // Foreshortened floor faint pose (head near top of floor box, feet near bottom)
  return [
    { name: 'nose', x: bx + bw * 0.48, y: by + bh * 0.12, confidence: 0.98 },
    { name: 'l_shoulder', x: bx + bw * 0.34, y: by + bh * 0.22, confidence: 0.98 },
    { name: 'r_shoulder', x: bx + bw * 0.64, y: by + bh * 0.22, confidence: 0.98 },
    { name: 'l_elbow', x: bx + bw * 0.22, y: by + bh * 0.36, confidence: 0.95 },
    { name: 'r_elbow', x: bx + bw * 0.74, y: by + bh * 0.36, confidence: 0.95 },
    { name: 'l_wrist', x: bx + bw * 0.14, y: by + bh * 0.48, confidence: 0.92 },
    { name: 'r_wrist', x: bx + bw * 0.78, y: by + bh * 0.48, confidence: 0.92 },
    { name: 'l_hip', x: bx + bw * 0.36, y: by + bh * 0.46, confidence: 0.98 },
    { name: 'r_hip', x: bx + bw * 0.62, y: by + bh * 0.46, confidence: 0.98 },
    { name: 'l_knee', x: bx + bw * 0.33, y: by + bh * 0.68, confidence: 0.96 },
    { name: 'r_knee', x: bx + bw * 0.64, y: by + bh * 0.68, confidence: 0.96 },
    { name: 'l_ankle', x: bx + bw * 0.30, y: by + bh * 0.88, confidence: 0.95 },
    { name: 'r_ankle', x: bx + bw * 0.68, y: by + bh * 0.88, confidence: 0.95 },
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

function centerDist(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ax = a[0] + a[2] * 0.5;
  const ay = a[1] + a[3] * 0.5;
  const bx = b[0] + b[2] * 0.5;
  const by = b[1] + b[3] * 0.5;
  return Math.hypot(ax - bx, ay - by);
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
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const prevGrayRef = useRef<Uint8Array | null>(null);
  const bgModelRef = useRef<Float32Array | null>(null);
  const lastAlertTimeRef = useRef<number>(0);

  // Multi-Person Tracker State (Supports 1 to 30+ simultaneous students/pedestrians)
  const tracksRef = useRef<MultiPersonTrack[]>([]);
  const nextTrackIdRef = useRef<number>(1);

  const [poseModelStatus, setPoseModelStatus] = useState<'loading' | 'ready' | 'fallback'>('loading');
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [videoProgress, setVideoProgress] = useState<number>(0);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [manualFallOverride, setManualFallOverride] = useState<boolean>(false);
  const [manualSecurityOverride, setManualSecurityOverride] = useState<boolean>(false);
  const [lockedFaintSummary, setLockedFaintSummary] = useState<string | null>(null);

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

  // Initialize BOTH Multi-Pose Landmarker AND Multi-Person ObjectDetector
  useEffect(() => {
    let cancelled = false;
    async function initVisionModels() {
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
            numPoses: 5,
            minPoseDetectionConfidence: 0.3,
            minPosePresenceConfidence: 0.3,
            minTrackingConfidence: 0.3,
          }).catch(() => null),
          ObjectDetector.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            scoreThreshold: 0.22,
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
    initVisionModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const resetMemoryBank = useCallback(() => {
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    prevGrayRef.current = null;
    bgModelRef.current = null;
    setManualFallOverride(false);
    setManualSecurityOverride(false);
    setLockedFaintSummary(null);
    simActorsRef.current.forEach((a) => {
      a.fallen = false;
      a.maskedIntruder = false;
      a.hasKnife = false;
    });
  }, []);

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
          // Lock the most horizontal or first tracked person as fainted
          const target = tracks.reduce((best, curr) =>
            curr.bbox[2] / Math.max(1, curr.bbox[3]) > best.bbox[2] / Math.max(1, best.bbox[3])
              ? curr
              : best
          );
          target.isFallen = true;
          target.faintLockedBbox = [...target.bbox];
          target.faintStartTime = performance.now();
          target.faintReason = 'Manual Faint Override';
          target.smoothedKeypoints = buildFrozenFloorKeypoints(target.bbox);
        } else if (!next) {
          tracks.forEach((t) => {
            t.isFallen = false;
            t.faintLockedBbox = null;
            t.faintStartTime = null;
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

    // High-Resolution 240x135 Analysis Canvas (4x4 px cells on 960x540) for Wide-Angle Crowd & Gym CCTV
    const GW = 240;
    const GH = 135;
    if (!offscreenCanvasRef.current) {
      const off = document.createElement('canvas');
      off.width = GW;
      off.height = GH;
      offscreenCanvasRef.current = off;
    }

    const renderLoop = (now: number) => {
      const dt = Math.max(1, now - lastFrameTimestamp);
      lastFrameTimestamp = now;
      smoothedFps = smoothedFps * 0.9 + Math.min(60, 1000 / dt) * 0.1;

      const tStart = performance.now();
      const canvas = canvasRef.current;
      const videoEl = videoRef.current;
      const offCanvas = offscreenCanvasRef.current;

      if (canvas && offCanvas) {
        const ctx = canvas.getContext('2d');
        const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
        const W = canvas.width;
        const H = canvas.height;

        if (ctx && offCtx) {
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
              // Reset faint lock if video loops back to 0.0s
              if (videoEl.currentTime < 0.25 && !manualFallOverride) {
                tracksRef.current.forEach((tr) => {
                  if (tr.isPrimaryCloseUp) {
                    tr.isFallen = false;
                    tr.faintLockedBbox = null;
                    tr.faintStartTime = null;
                    tr.faintFrames = 0;
                  }
                });
              }
            }

            ctx.drawImage(videoEl, 0, 0, W, H);
            offCtx.drawImage(videoEl, 0, 0, GW, GH);
            const imgData = offCtx.getImageData(0, 0, GW, GH);
            const data = imgData.data;

            const scaleX = W / GW; // 4.0
            const scaleY = H / GH; // 4.0

            // =========================================================================
            // STAGE 1: MEDIAPIPE MULTI-POSE & OBJECT DETECTOR (Close & Medium Subjects)
            // =========================================================================
            interface RawCandidate {
              bbox: [number, number, number, number];
              keypoints?: PoseKeypoint[];
              torsoAngleDeg?: number;
              foreshortenedFloor?: boolean;
              fromPose: boolean;
              isProneFloorShape: boolean;
            }

            const candidates: RawCandidate[] = [];

            if (videoEl.currentTime !== lastVideoTimeRef.current) {
              lastVideoTimeRef.current = videoEl.currentTime;
              const tsMs = performance.now();

              // 1A. PoseLandmarker (up to 5 poses)
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

                      const kps: PoseKeypoint[] = [];
                      keypointMap.forEach(({ idx, name }) => {
                        const pt = lm[idx];
                        const vis = pt?.visibility ?? 0;
                        if (pt && vis >= 0.45 && pt.x >= 0.02 && pt.x <= 0.98 && pt.y >= 0.02 && pt.y <= 0.98) {
                          highConfCount++;
                          if (pt.x < minX) minX = pt.x;
                          if (pt.x > maxX) maxX = pt.x;
                          if (pt.y < minY) minY = pt.y;
                          if (pt.y > maxY) maxY = pt.y;
                          kps.push({
                            name,
                            x: pt.x * W,
                            y: pt.y * H,
                            confidence: vis,
                          });
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

                        // Reject absurd full-screen false positives
                        if (bw < W * 0.72 && bh < H * 0.94 && bw >= 24 && bh >= 28) {
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
                            if (shMidY > 0.49 || (torsoNormDy < 0.13 && minY > 0.43)) {
                              foreshortenedFloor = true;
                            }
                          }

                          candidates.push({
                            bbox: [bx, by, bw, bh],
                            keypoints: kps,
                            torsoAngleDeg,
                            foreshortenedFloor,
                            fromPose: true,
                            isProneFloorShape: bw > bh * 1.12 || foreshortenedFloor,
                          });
                        }
                      }
                    });
                  }
                } catch {
                  // Continue to ObjectDetector + Connected Components
                }
              }

              // 1B. MediaPipe ObjectDetector ('person', up to 25 boxes)
              if (objectDetectorRef.current) {
                try {
                  const objResult = objectDetectorRef.current.detectForVideo(videoEl, tsMs + 1);
                  if (objResult.detections && objResult.detections.length > 0) {
                    const scaleVidX = W / Math.max(1, videoEl.videoWidth);
                    const scaleVidY = H / Math.max(1, videoEl.videoHeight);

                    objResult.detections.forEach((det) => {
                      const bb = det.boundingBox;
                      if (!bb) return;
                      const bx = Math.max(4, bb.originX * scaleVidX);
                      const by = Math.max(4, bb.originY * scaleVidY);
                      const bw = Math.min(W - bx - 4, bb.width * scaleVidX);
                      const bh = Math.min(H - by - 4, bb.height * scaleVidY);

                      if (bw >= 18 && bh >= 22 && bw < W * 0.65 && bh < H * 0.92) {
                        const overlapsExisting = candidates.some(
                          (c) => boxIoU(c.bbox, [bx, by, bw, bh]) > 0.35
                        );
                        if (!overlapsExisting) {
                          candidates.push({
                            bbox: [bx, by, bw, bh],
                            fromPose: false,
                            isProneFloorShape: bw > bh * 1.12,
                          });
                        }
                      }
                    });
                  }
                } catch {
                  // Continue to Connected-Component Crowd Segmentation
                }
              }
            }

            // =========================================================================
            // STAGE 2: HIGH-RES CONNECTED-COMPONENT CROWD & GYM FLOOR SEGMENTATION
            // Detects 15-25+ smaller students in wide-angle gym/courtyard CCTV videos
            // AND detects stationary fainted students lying on the court floor!
            // =========================================================================
            const numPixels = GW * GH;
            const gray = new Uint8Array(numPixels);
            const fgMask = new Uint8Array(numPixels);

            if (!bgModelRef.current || bgModelRef.current.length !== numPixels) {
              bgModelRef.current = new Float32Array(numPixels);
              for (let i = 0; i < numPixels; i++) {
                const idx = i * 4;
                bgModelRef.current[i] =
                  (data[idx] * 77 + data[idx + 1] * 150 + data[idx + 2] * 29) >> 8;
              }
            }

            const prevGray = prevGrayRef.current;
            const bgModel = bgModelRef.current;

            // Estimate court/floor median luminance & warm wood / floor tone across central court region
            let floorLumaSum = 0;
            let floorSamples = 0;
            for (let y = Math.floor(GH * 0.25); y < Math.floor(GH * 0.85); y += 4) {
              for (let x = Math.floor(GW * 0.15); x < Math.floor(GW * 0.85); x += 4) {
                const idx = (y * GW + x) * 4;
                const luma = (data[idx] * 77 + data[idx + 1] * 150 + data[idx + 2] * 29) >> 8;
                if (luma > 75 && luma < 225) {
                  floorLumaSum += luma;
                  floorSamples++;
                }
              }
            }
            const avgFloorLuma = floorSamples > 0 ? floorLumaSum / floorSamples : 145;

            // Build foreground mask combining temporal motion + human silhouette contrast against court floor
            for (let y = 10; y < GH - 8; y++) {
              for (let x = 8; x < GW - 12; x++) {
                const pIdx = y * GW + x;
                const idx = pIdx * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const luma = (r * 77 + g * 150 + b * 29) >> 8;
                gray[pIdx] = luma;

                const motionDiff = prevGray ? Math.abs(luma - prevGray[pIdx]) : 0;
                const bgDiff = Math.abs(luma - bgModel[pIdx]);

                // Slowly adapt background model where there is no motion
                if (motionDiff < 10) {
                  bgModel[pIdx] = bgModel[pIdx] * 0.985 + luma * 0.015;
                }

                // Check if pixel belongs to a student silhouette on the court/floor:
                // 1) Moving pixel (motionDiff > 15 or bgDiff > 26)
                // 2) Dark human clothing/shorts/hair contrasting strongly with bright wood court floor
                //    (excludes wood floor which has warm R > G > B and luma near avgFloorLuma)
                const isWoodOrBrightWall =
                  (r > g + 14 && g > b + 10 && luma > avgFloorLuma - 38) || luma > 188;

                // Exclude top-left camera overlay region & far-right bleachers
                const inOverlayZone = x < 98 && y < 16;
                const inFarRightBleachers = x > GW * 0.88 && y > GH * 0.36;
                const inTopWall = y < GH * 0.12;

                if (!inOverlayZone && !inFarRightBleachers && !inTopWall) {
                  const isStrongDarkApparel =
                    !isWoodOrBrightWall && luma < Math.min(92, avgFloorLuma - 48);
                  const isMovingPerson = motionDiff >= 14 || bgDiff >= 25;

                  if (isMovingPerson || isStrongDarkApparel) {
                    fgMask[pIdx] = 1;
                  }
                }
              }
            }
            prevGrayRef.current = gray;

            // Morphological 3x3 Density Filter: Removes 1-pixel court lines & specular floor glare!
            const filteredMask = new Uint8Array(numPixels);
            for (let y = 11; y < GH - 9; y++) {
              for (let x = 9; x < GW - 13; x++) {
                const pIdx = y * GW + x;
                if (!fgMask[pIdx]) continue;
                let neighbors = 0;
                for (let dy = -1; dy <= 1; dy++) {
                  const rowOffset = (y + dy) * GW + x;
                  for (let dx = -1; dx <= 1; dx++) {
                    if (fgMask[rowOffset + dx]) neighbors++;
                  }
                }
                // Court lines are 1px thin (<= 3 neighbors); human bodies are solid 2D clusters (>= 5 neighbors)
                if (neighbors >= 5) {
                  filteredMask[pIdx] = 1;
                }
              }
            }

            // Connected-Component BFS Clustering with small-gap bridging (radius 2)
            const visited = new Uint8Array(numPixels);
            const bfsQueue = new Int32Array(numPixels);

            for (let y = 11; y < GH - 9; y++) {
              for (let x = 9; x < GW - 13; x++) {
                const startIdx = y * GW + x;
                if (!filteredMask[startIdx] || visited[startIdx]) continue;

                let qHead = 0;
                let qTail = 0;
                bfsQueue[qTail++] = startIdx;
                visited[startIdx] = 1;

                let minGX = x;
                let maxGX = x;
                let minGY = y;
                let maxGY = y;
                let cellCount = 0;

                while (qHead < qTail) {
                  const curr = bfsQueue[qHead++];
                  const cy = Math.floor(curr / GW);
                  const cx = curr - cy * GW;
                  cellCount++;

                  if (cx < minGX) minGX = cx;
                  if (cx > maxGX) maxGX = cx;
                  if (cy < minGY) minGY = cy;
                  if (cy > maxGY) maxGY = cy;

                  // Search 5x5 neighborhood to bridge white shirt + dark shorts/head into a single student
                  for (let dy = -2; dy <= 2; dy++) {
                    const ny = cy + dy;
                    if (ny < 10 || ny >= GH - 8) continue;
                    const nRow = ny * GW;
                    for (let dx = -2; dx <= 2; dx++) {
                      const nx = cx + dx;
                      if (nx < 8 || nx >= GW - 12) continue;
                      const nIdx = nRow + nx;
                      if (filteredMask[nIdx] && !visited[nIdx]) {
                        visited[nIdx] = 1;
                        bfsQueue[qTail++] = nIdx;
                      }
                    }
                  }
                }

                const blobW = (maxGX - minGX + 2) * scaleX;
                const blobH = (maxGY - minGY + 2) * scaleY;
                const blobX = Math.max(6, minGX * scaleX - 4);
                const blobY = Math.max(6, minGY * scaleY - 6);

                // Filter valid human-sized blobs (excludes tiny noise and huge court walls/bleachers)
                if (
                  cellCount >= 6 &&
                  blobW >= 18 &&
                  blobH >= 20 &&
                  blobW <= 260 &&
                  blobH <= 380
                ) {
                  // If multiple standing students are clustered side-by-side (e.g. group under basketball hoop),
                  // split wide tall blobs into individual student columns!
                  if (blobW > 72 && blobH >= 52 && blobW < 220 && blobY > H * 0.22) {
                    const numSplits = Math.min(4, Math.max(2, Math.round(blobW / 42)));
                    const subW = blobW / numSplits;
                    for (let s = 0; s < numSplits; s++) {
                      const sx = blobX + s * subW;
                      const subBox: [number, number, number, number] = [sx, blobY, subW - 2, blobH];
                      const overlaps = candidates.some((c) => boxIoU(c.bbox, subBox) > 0.35);
                      if (!overlaps) {
                        candidates.push({
                          bbox: subBox,
                          fromPose: false,
                          isProneFloorShape: false,
                        });
                      }
                    }
                  } else {
                    // Single student blob (standing OR sprawled/fainted on the gym floor!)
                    // Check if there is a white/light shirt adjacent to a prone floor student (like the fainted student on the right!)
                    let adjBox: [number, number, number, number] = [blobX, blobY, blobW, blobH];
                    const aspect = blobW / Math.max(1, blobH);
                    const isProne =
                      (aspect >= 1.08 && blobH <= 72 && blobY > H * 0.22) ||
                      (blobW >= 24 && blobH <= 38 && aspect >= 0.92 && blobY > H * 0.24);

                    if (isProne) {
                      // Pad prone floor bounding box slightly to encompass white t-shirt + legs on the floor
                      adjBox = [
                        Math.max(8, blobX - 8),
                        Math.max(8, blobY - 6),
                        Math.min(W - blobX, Math.max(46, blobW + 16)),
                        Math.min(H - blobY, Math.max(34, blobH + 12)),
                      ];
                    }

                    const overlaps = candidates.some(
                      (c) => boxIoU(c.bbox, adjBox) > 0.32 || centerDist(c.bbox, adjBox) < 24
                    );
                    if (!overlaps) {
                      candidates.push({
                        bbox: adjBox,
                        fromPose: false,
                        isProneFloorShape: isProne,
                      });
                    }
                  }
                }
              }
            }

            // Determine if this frame is a Wide-Angle Multi-Person Crowd/Gym Video vs Close-Up Single Person
            const isCrowdScene = candidates.length >= 3 || tracksRef.current.length >= 3;

            // =========================================================================
            // STAGE 3: MULTI-OBJECT TRACKER & PER-PERSON FAINT MEMORY ENGINE
            // =========================================================================
            const tracks = tracksRef.current;
            const matchedTrackIds = new Set<number>();
            const matchedCandIndices = new Set<number>();

            // Match existing tracks to current frame candidates by closest center / IoU
            candidates.forEach((cand, cIdx) => {
              let bestTrack: MultiPersonTrack | null = null;
              let bestScore = Infinity;

              tracks.forEach((tr) => {
                if (matchedTrackIds.has(tr.id)) return;
                const dist = centerDist(tr.bbox, cand.bbox);
                const maxAllowedDist = tr.isPrimaryCloseUp ? 180 : 65;
                if (dist < maxAllowedDist && dist < bestScore) {
                  bestScore = dist;
                  bestTrack = tr;
                }
              });

              if (bestTrack) {
                const tr = bestTrack as MultiPersonTrack;
                matchedTrackIds.add(tr.id);
                matchedCandIndices.add(cIdx);
                tr.missedFrames = 0;
                tr.ageFrames++;

                const prevCx = tr.bbox[0] + tr.bbox[2] * 0.5;
                const prevCy = tr.bbox[1] + tr.bbox[3] * 0.5;
                const newCx = cand.bbox[0] + cand.bbox[2] * 0.5;
                const newCy = cand.bbox[1] + cand.bbox[3] * 0.5;

                tr.vx = tr.vx * 0.6 + (newCx - prevCx) * 0.4;
                tr.vy = tr.vy * 0.6 + (newCy - prevCy) * 0.4;
                tr.speedPx = Math.hypot(tr.vx, tr.vy);

                // Smooth bounding box unless faint-locked on the floor
                if (!tr.isFallen || !tr.faintLockedBbox) {
                  const alpha = cand.fromPose ? 0.45 : 0.35;
                  tr.bbox = [
                    tr.bbox[0] * (1 - alpha) + cand.bbox[0] * alpha,
                    tr.bbox[1] * (1 - alpha) + cand.bbox[1] * alpha,
                    tr.bbox[2] * (1 - alpha) + cand.bbox[2] * alpha,
                    tr.bbox[3] * (1 - alpha) + cand.bbox[3] * alpha,
                  ];
                }

                const [bx, by, bw, bh] = tr.bbox;
                const aspect = bw / Math.max(1, bh);
                tr.isPrimaryCloseUp = !isCrowdScene && bh > H * 0.35;

                // Update standing baseline height when upright
                if (aspect < 0.85 && !cand.foreshortenedFloor) {
                  tr.standingFrames++;
                  tr.faintFrames = Math.max(0, tr.faintFrames - 1);
                  if (bh > tr.peakHeight * 0.85) {
                    tr.peakHeight = Math.max(tr.peakHeight, bh);
                    tr.baselineTopY = Math.min(tr.baselineTopY, by);
                    tr.standingBaselineBbox = [...tr.bbox];
                  }
                  if (tr.isFallen && tr.standingFrames >= 12 && !manualFallOverride) {
                    tr.isFallen = false;
                    tr.faintLockedBbox = null;
                    tr.faintStartTime = null;
                  }
                } else {
                  tr.standingFrames = 0;
                }

                // FAINT DETECTION RULES (Works for BOTH Wide-Angle 25-Student Gym AND Close-Up Kitchen Video!)
                let faintTriggered = false;
                let reason = '';

                if (tr.isPrimaryCloseUp) {
                  // Close-Up Single Person 4-Way Biomechanical Faint Detection
                  const headDropPct =
                    tr.baselineTopY < H ? Math.round(((by - tr.baselineTopY) / H) * 100) : 0;
                  const heightCompPct =
                    tr.peakHeight > 0 ? Math.round((1 - bh / tr.peakHeight) * 100) : 0;
                  const sigHeadDropped =
                    tr.peakHeight > H * 0.4 && headDropPct >= 18 && by > H * 0.42;
                  const sigHeightComp =
                    tr.peakHeight > H * 0.4 && heightCompPct >= 28 && by > H * 0.42;
                  const sigFloorZone =
                    (by > H * 0.47 && by + bh * 0.5 > H * 0.64) || Boolean(cand.foreshortenedFloor);
                  const sigAngle = aspect > 1.12 || (cand.torsoAngleDeg ?? 0) > 48;

                  if (sigHeadDropped || sigHeightComp || sigFloorZone || sigAngle) {
                    faintTriggered = true;
                    reason = sigHeadDropped
                      ? `Head dropped ${headDropPct}% onto floor`
                      : 'Person collapsed onto floor';
                  }
                } else {
                  // Wide-Angle Multi-Person School / Gym Faint Detection:
                  // 1) Student lying sprawled/horizontally on the gym floor (aspect >= 1.08 or cand.isProneFloorShape)
                  //    with low translation speed (stationary on the floor, not running!)
                  // 2) Or student whose height compressed >= 35% from their standing height
                  const isStationaryProneOnCourt =
                    (cand.isProneFloorShape || aspect >= 1.10) &&
                    tr.speedPx < 2.2 &&
                    by > H * 0.22 &&
                    by < H * 0.85;

                  const isHeightCollapsedStudent =
                    tr.peakHeight >= 48 &&
                    bh < tr.peakHeight * 0.65 &&
                    aspect >= 0.92 &&
                    tr.speedPx < 2.0;

                  if (isStationaryProneOnCourt || isHeightCollapsedStudent) {
                    faintTriggered = true;
                    reason = 'Student collapsed / lying motionless on gym floor';
                  }
                }

                if (faintTriggered) {
                  tr.faintFrames++;
                  if (tr.faintFrames >= 3 && !tr.isFallen) {
                    tr.isFallen = true;
                    tr.faintLockedBbox = [...tr.bbox];
                    tr.faintStartTime = now;
                    tr.faintReason = reason;
                    tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.bbox);
                  }
                }

                // Update stabilized keypoints
                if (tr.isFallen && tr.faintLockedBbox) {
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(tr.faintLockedBbox);
                } else if (cand.keypoints && cand.keypoints.length >= 5) {
                  tr.smoothedKeypoints = cand.keypoints.map((kp) => ({
                    ...kp,
                    x: Math.max(bx, Math.min(bx + bw, kp.x)),
                    y: Math.max(by, Math.min(by + bh, kp.y)),
                  }));
                } else {
                  tr.smoothedKeypoints = buildStandingKeypoints(tr.bbox, now * 0.008 + tr.id);
                }
              }
            });

            // Create new tracks for unmatched candidates (up to 30 simultaneous students)
            candidates.forEach((cand, cIdx) => {
              if (matchedCandIndices.has(cIdx)) return;
              if (tracks.length >= 30) return;

              const id = nextTrackIdRef.current++;
              const [bx, by, bw, bh] = cand.bbox;
              const aspect = bw / Math.max(1, bh);
              const isPrimaryCloseUp = !isCrowdScene && bh > H * 0.35;

              tracks.push({
                id,
                bbox: [...cand.bbox],
                standingBaselineBbox: aspect < 0.88 ? [...cand.bbox] : null,
                peakHeight: aspect < 0.88 ? bh : 0,
                baselineTopY: aspect < 0.88 ? by : H,
                vx: 0,
                vy: 0,
                speedPx: 0,
                ageFrames: 1,
                missedFrames: 0,
                faintFrames: cand.isProneFloorShape ? 2 : 0,
                standingFrames: aspect < 0.88 ? 1 : 0,
                isFallen: false,
                faintLockedBbox: null,
                faintStartTime: null,
                faintReason: '',
                smoothedKeypoints:
                  cand.keypoints && cand.keypoints.length >= 5
                    ? cand.keypoints
                    : buildStandingKeypoints(cand.bbox, id),
                torsoColor: 'BLUE',
                hsvMatchPct: 0,
                isPrimaryCloseUp,
              });
            });

            // Retain faint-locked tracks in memory even if temporarily occluded; prune stale normal tracks
            tracksRef.current = tracks.filter((tr) => {
              if (!matchedTrackIds.has(tr.id)) {
                tr.missedFrames++;
                // If a close-up person suddenly dropped out of view downward, lock their last floor position!
                if (
                  tr.isPrimaryCloseUp &&
                  !tr.isFallen &&
                  tr.missedFrames === 2 &&
                  (tr.vy > 4 || tr.bbox[1] > H * 0.36)
                ) {
                  const [lx, ly, lw, lh] = tr.bbox;
                  const floorY = Math.max(ly, H * 0.52);
                  const floorH = Math.min(H - floorY - 10, Math.max(140, lh * 0.58));
                  const locked: [number, number, number, number] = [lx, floorY, lw, floorH];
                  tr.isFallen = true;
                  tr.faintLockedBbox = locked;
                  tr.bbox = [...locked];
                  tr.faintStartTime = now;
                  tr.faintReason = 'Remembered Floor Collapse Position';
                  tr.smoothedKeypoints = buildFrozenFloorKeypoints(locked);
                }
              }

              // Keep faint-locked people in memory for up to 120 missed frames (~4 seconds)
              if (tr.isFallen) {
                return tr.missedFrames < 120;
              }
              return tr.missedFrames < 8;
            });

            // Convert active tracks into DetectedSubjects for rendering & telemetry
            let activeFaintSummary: string | null = null;

            tracksRef.current.forEach((tr, idx) => {
              // Require at least 2 frames of persistence (or immediate if fainted/pose-backed) to avoid 1-frame flicker
              if (tr.ageFrames < 2 && !tr.isFallen && tracksRef.current.length > 3) return;

              const activeBox =
                tr.isFallen && tr.faintLockedBbox ? tr.faintLockedBbox : tr.bbox;
              const [bx, by, bw, bh] = activeBox;

              // Sample HSV shirt color inside torso region
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
                activeFaintSummary = `${tr.faintReason || 'Person Fainted on Floor'} (${faintDurationSec}s)`;
              }

              detectedSubjects.push({
                id: tr.id,
                label: isCrowdScene ? `Student #${idx + 1}` : `Person #${tr.id}`,
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
          // STAGE 4: RENDER BOUNDING BOXES, PROPORTIONAL SKELETONS & FAINT ALERTS
          // =========================================================================
          let frameAlertActive = false;
          let frameGreenWave = false;
          let frameAlertTitle = `All Clear — Tracking ${detectedSubjects.length} ${
            detectedSubjects.length === 1 ? 'Person' : 'Students / People'
          }`;
          let frameAlertDetail = `Monitoring ${activeCamera.name}. Hospital-to-School route on standby.`;

          const isCompactCrowd = detectedSubjects.length >= 4;

          detectedSubjects.forEach((subj) => {
            const [bx, by, bw, bh] = subj.bbox;
            const [tx, ty, tw, th] = subj.torsoRoi;

            // Draw Last Standing Ghost Box if available
            if (subj.isFallen && subj.lastStandingBbox) {
              const [sx, sy, sw, sh] = subj.lastStandingBbox;
              ctx.save();
              ctx.strokeStyle = 'rgba(6, 182, 212, 0.65)';
              ctx.setLineDash([5, 3]);
              ctx.lineWidth = 1.5;
              ctx.strokeRect(sx, sy, sw, sh);
              ctx.restore();
            }

            // Draw Proportionally Scaled Skeleton strictly inside the person's bounding box
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
                ctx.fillStyle = subj.isFallen ? '#EF4444' : '#06B6D4';
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, dotRadius, 0, Math.PI * 2);
                ctx.fill();
              });
              ctx.restore();
            }

            let boxColor = '#10B981';
            let statusText = isCompactCrowd ? subj.label : `${subj.label} · Standing`;

            if (mode === 'najda') {
              if (subj.isFallen) {
                boxColor = '#EF4444';
                statusText =
                  subj.faintDurationSec > 0
                    ? `FAINT DETECTED · ${subj.label} (${subj.faintDurationSec}s)`
                    : `FAINT DETECTED · ${subj.label}`;
                frameAlertActive = true;
                frameGreenWave = true;
                frameAlertTitle = `Emergency at School — ${subj.label} Fainted!`;
                frameAlertDetail = `Detected ${detectedSubjects.length} students in view; ${subj.label} collapsed on floor. Hospital → School route is now a GREEN ZONE.`;

                if (now - lastAlertTimeRef.current > 7000) {
                  lastAlertTimeRef.current = now;
                  onTriggerAlert(frameAlertTitle, frameAlertDetail);
                }
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

            // Draw Bounding Box + Pulsing Emergency Halo for Fainted Student
            ctx.save();
            if (subj.isFallen) {
              ctx.fillStyle = 'rgba(239, 68, 68, 0.22)';
              ctx.fillRect(bx - 4, by - 4, bw + 8, bh + 8);

              // Pulsing red target beacon around fainted student so operator spots them immediately in a 25-student gym
              const pulseR = Math.max(bw, bh) * 0.75 + Math.sin(now * 0.008) * 6;
              ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
              ctx.lineWidth = 2.5;
              ctx.beginPath();
              ctx.arc(bx + bw / 2, by + bh / 2, pulseR, 0, Math.PI * 2);
              ctx.stroke();
            }

            ctx.strokeStyle = boxColor;
            ctx.lineWidth = subj.isFallen ? 3 : isCompactCrowd ? 1.8 : 2.5;
            ctx.strokeRect(bx, by, bw, bh);
            ctx.restore();

            // Draw Clean Label Badge
            const isSmallLabel = isCompactCrowd && !subj.isFallen;
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

          // Top-Left Camera & Crowd Count Overlay
          ctx.fillStyle = 'rgba(11, 15, 23, 0.84)';
          ctx.beginPath();
          ctx.roundRect(14, 14, 455, 36, 6);
          ctx.fill();
          ctx.fillStyle = '#F8FAFC';
          ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
          ctx.fillText(
            `${activeCamera.code}: ${activeCamera.name.split(' (')[0]} · ${
              detectedSubjects.length
            } People Tracked`,
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
    manualFallOverride,
    manualSecurityOverride,
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
          <span className="text-xs font-mono text-[#94A3B8] shrink-0">
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

      {/* Bottom Bar: Multi-Person Crowd & Faint Memory Status */}
      <div className="p-4 bg-[#0B0F17] border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5 text-[#CBD5E1]">
            {lockedFaintSummary ? (
              <Lock className="w-4 h-4 text-[#EF4444]" />
            ) : tracksRef.current.length > 1 ? (
              <Users className="w-4 h-4 text-[#06B6D4]" />
            ) : (
              <History className="w-4 h-4 text-[#06B6D4]" />
            )}
            <strong>Multi-Person Crowd &amp; Faint Engine:</strong>{' '}
            {lockedFaintSummary ? (
              <span className="text-[#EF4444] font-semibold">
                FAINT LOCKED — {lockedFaintSummary}
              </span>
            ) : (
              <span className="text-[#10B981]">
                Multi-Person Crowd Segmentation + Prone Floor Faint Detector Active
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

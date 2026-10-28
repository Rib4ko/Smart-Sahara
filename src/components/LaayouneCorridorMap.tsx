import React, { useEffect, useState, useRef } from 'react';
import {
  Navigation,
  Siren,
  MapPin,
  ShieldCheck,
  OctagonAlert,
  Play,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Shuffle,
  Crosshair,
} from 'lucide-react';
import {
  LAAYOUNE_CAMERA_NODES,
  CameraSectorNode,
} from '../types/sahra';

interface LaayouneCorridorMapProps {
  greenWaveActive: boolean;
  activeCameraId: string;
  onSelectCamera: (camera: CameraSectorNode) => void;
  onToggleManualGreenWave: () => void;
}

interface SimCar {
  id: string;
  routeId: string;
  isGreenCorridorDirection: boolean;
  progress: number;
  baseSpeed: number;
  jitterPhase: number;
  sizeScale: number;
  laneOffset: number;
  color: string;
}

// Interpolate [x, y, angleDeg] along any multi-turn polyline path with optional perpendicular lane offset
function getPointOnPolyline(
  points: [number, number][],
  t: number,
  laneOffset = 0
): { x: number; y: number; angle: number } {
  const clamped = ((t % 1) + 1) % 1;
  if (points.length < 2) return { x: points[0][0], y: points[0][1], angle: 0 };

  const segLengths: number[] = [];
  let totalLen = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const dx = points[i + 1][0] - points[i][0];
    const dy = points[i + 1][1] - points[i][1];
    const len = Math.hypot(dx, dy);
    segLengths.push(len);
    totalLen += len;
  }

  let targetDist = clamped * totalLen;
  for (let i = 0; i < segLengths.length; i++) {
    if (targetDist <= segLengths[i] || i === segLengths.length - 1) {
      const segLen = Math.max(0.0001, segLengths[i]);
      const localT = Math.max(0, Math.min(1, targetDist / segLen));
      const dx = points[i + 1][0] - points[i][0];
      const dy = points[i + 1][1] - points[i][1];
      const rad = Math.atan2(dy, dx);
      const nx = -Math.sin(rad) * laneOffset;
      const ny = Math.cos(rad) * laneOffset;

      const x = points[i][0] + dx * localT + nx;
      const y = points[i][1] + dy * localT + ny;
      const angle = (rad * 180) / Math.PI;
      return { x, y, angle };
    }
    targetDist -= segLengths[i];
  }

  const last = points[points.length - 1];
  return { x: last[0], y: last[1], angle: 0 };
}

function toSvgPoints(pts: [number, number][]): string {
  return pts.map(([x, y]) => `${x},${y}`).join(' ');
}

// Generate points around a circle for roundabout circulation
function createCirclePath(
  cx: number,
  cy: number,
  radius: number,
  steps = 24,
  clockwise = true
): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const ratio = i / steps;
    const angle = (clockwise ? 1 : -1) * ratio * Math.PI * 2;
    pts.push([
      Math.round((cx + Math.cos(angle) * radius) * 10) / 10,
      Math.round((cy + Math.sin(angle) * radius) * 10) / 10,
    ]);
  }
  return pts;
}

// ============================================================================
// REALISTIC ORGANIC LAÂYOUNE CITY ROAD NETWORK (1080 x 620 ViewBox)
// Non-orthogonal diagonal boulevards, roundabouts, irregular urban blocks,
// and Place du Méchouar rendered as a large empty circle.
// ============================================================================

export const PLACE_MECHOUAR_CENTER: [number, number] = [825, 165];
export const PLACE_MECHOUAR_RADIUS = 58; // Large prominent empty circle

export const HOSPITAL_ORIGIN: [number, number] = [115, 495];

// 1. MULTI-TURN EMERGENCY ROUTE (Hôpital Moulay Hassan -> 5 Realistic Organic Turns -> Place du Méchouar Circle)
const HOSPITAL_TO_MECHOUAR_PATH: [number, number][] = [
  [115, 495], // Hôpital Moulay Hassan Gate
  [255, 445], // Turn 1: Rond-Point Kairouan / Av. Hassan II
  [305, 290], // Turn 2: Carrefour Av. Hassan II & Boulevard Mekka
  [515, 325], // Turn 3: Rond-Point Al Massira
  [610, 185], // Turn 4: Carrefour Rue Smara & Av. Mohammed V
  [730, 212], // Turn 5: Approche Sud Place du Méchouar
  [770, 182], // Entry onto Place du Méchouar Empty Circle Perimeter
];

// 2. OPPOSITE LANE ALONG THE SAME MULTI-TURN CORRIDOR (Place du Méchouar -> Hospital, turns RED & STOPS on faint)
const OPPOSITE_CORRIDOR_PATH: [number, number][] = [
  [764, 168],
  [725, 198],
  [600, 171],
  [505, 311],
  [292, 276],
  [242, 433],
  [110, 481],
];

// Circular ring path around Place du Méchouar empty circle
const MECHOUAR_RING_PATH = createCirclePath(
  PLACE_MECHOUAR_CENTER[0],
  PLACE_MECHOUAR_CENTER[1],
  PLACE_MECHOUAR_RADIUS,
  28,
  false
);

// 3. ORGANIC CITY ROUTES (Angled avenues, diagonal streets & roundabouts across Laâyoune)
const CITY_ROUTES: Record<string, [number, number][]> = {
  'green-corridor': HOSPITAL_TO_MECHOUAR_PATH,
  'opp-corridor': OPPOSITE_CORRIDOR_PATH,
  'mechouar-ring': MECHOUAR_RING_PATH,

  // Diagonal Avenue Hassan II (North-West to South-East across Turn 2 & Turn 1)
  'av-hassan-nw-se': [
    [355, 35],
    [330, 155],
    [305, 290],
    [255, 445],
    [215, 585],
  ],

  // Diagonal Boulevard Mekka (West-North-West to East-South-East across Turn 2 & Turn 3)
  'bd-mekka-we': [
    [35, 240],
    [175, 265],
    [305, 290],
    [515, 325],
    [715, 355],
    [1040, 405],
  ],

  // Angled Avenue Al Massira (South-West to North-East across Turn 3 & Turn 4)
  'av-massira-sn': [
    [415, 585],
    [465, 455],
    [515, 325],
    [610, 185],
    [695, 35],
  ],

  // Rue Smara Winding Collector (West to North-East across Av. Hassan II, Turn 4 & Place du Méchouar)
  'rue-smara-we': [
    [45, 115],
    [195, 140],
    [330, 155],
    [475, 168],
    [610, 185],
    [730, 212],
    [770, 182],
  ],

  // Boulevard Kairouan Southern Arterial (East to West-South-West across Turn 1)
  'bd-kairouan-ew': [
    [1035, 525],
    [820, 495],
    [640, 475],
    [465, 455],
    [255, 445],
    [115, 495],
    [35, 525],
  ],

  // Avenue Mohammed V Eastern Connector (South to Place du Méchouar Circle to North)
  'av-mohammed-v': [
    [895, 585],
    [820, 495],
    [715, 355],
    [730, 212],
    [778, 198],
  ],

  // Northern Corniche / Oued Sakia Connector into Place du Méchouar
  'av-sakia-north': [
    [1040, 85],
    [878, 140],
  ],

  // Western Quartier Al Qods Diagonal Loop
  'quartier-qods-loop': [
    [175, 265],
    [195, 140],
    [330, 155],
    [305, 290],
    [175, 265],
  ],
};

// Secondary organic neighborhood streets (non-orthogonal realistic urban tissue)
const SECONDARY_STREETS: [number, number][][] = [
  [[195, 35], [195, 140], [175, 265], [135, 385], [115, 495]],
  [[475, 35], [475, 168], [415, 308], [365, 448], [335, 585]],
  [[715, 355], [640, 475], [595, 585]],
  [[883, 165], [1040, 185]],
  [[825, 223], [845, 375], [820, 495]],
  [[825, 107], [810, 30]],
  [[610, 185], [715, 355]],
  [[35, 375], [135, 385], [275, 380], [415, 308]],
  [[695, 35], [775, 120]],
  [[845, 375], [1040, 315]],
];

// Irregular realistic city block polygons (looking like a real satellite/vector city map)
const IRREGULAR_CITY_BLOCKS: string[] = [
  // NW Quarter
  '55,45 180,45 180,125 55,102',
  '210,45 338,45 316,140 210,126',
  '368,45 460,45 460,152 345,142',
  '490,45 675,45 598,168 490,154',
  // Upper-West & Central-West
  '50,130 180,152 162,250 48,225',
  '208,155 314,168 292,272 190,252',
  '342,170 462,182 405,292 318,276',
  '488,185 592,196 505,308 426,295',
  // Around Place du Méchouar & NE
  '628,182 745,120 755,165 718,195',
  '895,95 1025,52 1025,168 898,152',
  '895,182 1025,200 1025,298 858,352 842,235',
  // Mid-City South of Bd. Mekka
  '48,258 160,278 125,368 45,358',
  '186,282 288,302 265,368 148,370',
  '316,308 402,322 355,432 272,428',
  '425,326 498,338 452,438 375,434',
  '530,342 695,368 628,458 482,440',
  '622,212 715,226 702,338 625,295',
  '745,230 812,238 828,358 730,342',
  // Southern Districts
  '135,462 242,460 202,570 112,568',
  '272,460 352,462 322,570 228,570',
  '378,464 452,468 402,570 348,570',
  '482,470 622,486 580,572 432,572',
  '655,490 802,508 775,575 612,572',
  '732,374 828,386 806,478 655,462',
  '858,388 1020,335 1022,388 845,475',
];

const CAR_COLORS = [
  '#38BDF8',
  '#FBBF24',
  '#E2E8F0',
  '#F8FAFC',
  '#A78BFA',
  '#FB923C',
  '#94A3B8',
  '#2DD4BF',
];

function createRandomizedCars(): SimCar[] {
  const specs: Array<{ routeId: string; isGreen: boolean; count: number }> = [
    { routeId: 'green-corridor', isGreen: true, count: 3 },
    { routeId: 'opp-corridor', isGreen: false, count: 3 },
    { routeId: 'mechouar-ring', isGreen: false, count: 3 },
    { routeId: 'av-hassan-nw-se', isGreen: false, count: 2 },
    { routeId: 'bd-mekka-we', isGreen: false, count: 3 },
    { routeId: 'av-massira-sn', isGreen: false, count: 2 },
    { routeId: 'rue-smara-we', isGreen: false, count: 2 },
    { routeId: 'bd-kairouan-ew', isGreen: false, count: 2 },
    { routeId: 'av-mohammed-v', isGreen: false, count: 2 },
    { routeId: 'quartier-qods-loop', isGreen: false, count: 2 },
  ];

  const list: SimCar[] = [];
  let idx = 1;
  specs.forEach(({ routeId, isGreen, count }) => {
    for (let i = 0; i < count; i++) {
      const baseProgress = (i / count + Math.random() * 0.22) % 1;
      list.push({
        id: `car-${idx++}`,
        routeId,
        isGreenCorridorDirection: isGreen,
        progress: baseProgress,
        baseSpeed: 0.0016 + Math.random() * 0.0017,
        jitterPhase: Math.random() * Math.PI * 2,
        sizeScale: 0.85 + Math.random() * 0.32,
        laneOffset: (Math.random() - 0.5) * 4.5,
        color: CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)],
      });
    }
  });
  return list;
}

// The 5 Organic Turn Intersections along the Hospital -> Place du Méchouar Route
const CORRIDOR_TURN_INTERSECTIONS = [
  { id: 'turn-1', label: 'Turn 1: Rond-Point Kairouan', x: 255, y: 445, isRoundabout: true },
  { id: 'turn-2', label: 'Turn 2: Av. Hassan II × Bd. Mekka', x: 305, y: 290, isRoundabout: false },
  { id: 'turn-3', label: 'Turn 3: Rond-Point Al Massira', x: 515, y: 325, isRoundabout: true },
  { id: 'turn-4', label: 'Turn 4: Rue Smara × Av. Med V', x: 610, y: 185, isRoundabout: false },
  { id: 'turn-5', label: 'Turn 5: Entrée Place du Méchouar', x: 730, y: 212, isRoundabout: false },
];

export const LaayouneCorridorMap: React.FC<LaayouneCorridorMapProps> = ({
  greenWaveActive,
  activeCameraId,
  onSelectCamera,
  onToggleManualGreenWave,
}) => {
  const [cars, setCars] = useState<SimCar[]>(() => createRandomizedCars());
  const [ambulanceProgress, setAmbulanceProgress] = useState<number>(0);

  // Interactive Zoom & Pan State
  const [zoom, setZoom] = useState<number>(1.15);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [followAmbulance, setFollowAmbulance] = useState<boolean>(false);
  const dragStartRef = useRef<{ clientX: number; clientY: number; panX: number; panY: number } | null>(
    null
  );

  const greenWaveRef = useRef<boolean>(greenWaveActive);

  useEffect(() => {
    greenWaveRef.current = greenWaveActive;
    if (!greenWaveActive) {
      setAmbulanceProgress(0);
      setFollowAmbulance(false);
    }
  }, [greenWaveActive]);

  // 30 FPS Organic City Traffic & Multi-Turn Ambulance Tracker Loop
  useEffect(() => {
    let tick = 0;
    const interval = setInterval(() => {
      tick++;
      const isEmergency = greenWaveRef.current;

      setCars((prevCars) =>
        prevCars.map((car) => {
          // When person faints, all cars on opposite & cross routes STOP at the red lights
          if (isEmergency && !car.isGreenCorridorDirection) {
            return car;
          }

          // Organic random speed variation so cars don't move robotically
          const wave = 1 + Math.sin(tick * 0.05 + car.jitterPhase) * 0.25;
          const boost = isEmergency && car.isGreenCorridorDirection ? 1.65 : wave;
          let nextProg = car.progress + car.baseSpeed * boost;
          if (nextProg > 1) nextProg -= 1;

          return {
            ...car,
            progress: nextProg,
          };
        })
      );

      if (isEmergency) {
        setAmbulanceProgress((prev) => {
          if (prev >= 1) return 0;
          return Math.min(1, prev + 0.0033);
        });
      }
    }, 35);

    return () => clearInterval(interval);
  }, []);

  const ambulanceCoord = getPointOnPolyline(
    HOSPITAL_TO_MECHOUAR_PATH,
    ambulanceProgress
  );
  const remainingKm = Math.max(0, (1 - ambulanceProgress) * 3.4).toFixed(2);
  const etaSeconds = Math.max(0, Math.ceil((1 - ambulanceProgress) * 120));

  const currentTurnStatus =
    ambulanceProgress < 0.16
      ? "Rue de l'Hôpital → Approaching Rond-Point Kairouan (Turn 1)"
      : ambulanceProgress < 0.35
      ? 'Turn 1 Left → North-East on Av. Hassan II'
      : ambulanceProgress < 0.58
      ? 'Turn 2 Right → Diagonal on Boulevard Mekka'
      : ambulanceProgress < 0.75
      ? 'Turn 3 Left → Rond-Point Al Massira'
      : ambulanceProgress < 0.90
      ? 'Turn 4 Right → Rue Smara / Av. Mohammed V'
      : ambulanceProgress < 0.98
      ? 'Turn 5 Left → Entering Place du Méchouar Circle'
      : 'Arrived at Place du Méchouar (Laâyoune)';

  // Dynamic SVG ViewBox calculation for smooth Zoom & Pan
  const baseW = 1080;
  const baseH = 620;
  const viewW = baseW / zoom;
  const viewH = baseH / zoom;

  let centerX = baseW / 2 - pan.x;
  let centerY = baseH / 2 - pan.y;

  if (followAmbulance && greenWaveActive) {
    centerX = ambulanceCoord.x;
    centerY = ambulanceCoord.y;
  }

  const clampedCenterX = Math.max(viewW / 2, Math.min(baseW - viewW / 2, centerX));
  const clampedCenterY = Math.max(viewH / 2, Math.min(baseH - viewH / 2, centerY));
  const viewBoxX = Math.round(clampedCenterX - viewW / 2);
  const viewBoxY = Math.round(clampedCenterY - viewH / 2);

  const handleZoomIn = () => {
    setZoom((z) => Math.min(3.2, Math.round((z + 0.35) * 100) / 100));
  };

  const handleZoomOut = () => {
    setZoom((z) => {
      const next = Math.max(1.0, Math.round((z - 0.35) * 100) / 100);
      if (next === 1.0) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetView = () => {
    setFollowAmbulance(false);
    setZoom(1.15);
    setPan({ x: 0, y: 0 });
  };

  const handleFocusPlaceMechouar = () => {
    setFollowAmbulance(false);
    setZoom(2.1);
    setPan({
      x: baseW / 2 - PLACE_MECHOUAR_CENTER[0] + 45,
      y: baseH / 2 - PLACE_MECHOUAR_CENTER[1] - 25,
    });
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (followAmbulance) setFollowAmbulance(false);
    setIsDragging(true);
    dragStartRef.current = {
      clientX: e.clientX,
      clientY: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDragging || !dragStartRef.current) return;
    const dx = (e.clientX - dragStartRef.current.clientX) * (1.2 / zoom);
    const dy = (e.clientY - dragStartRef.current.clientY) * (1.2 / zoom);
    setPan({
      x: dragStartRef.current.panX + dx,
      y: dragStartRef.current.panY + dy,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    dragStartRef.current = null;
  };

  const handleWheelZoom = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoom((z) => Math.min(3.2, Math.round((z + 0.2) * 100) / 100));
    } else {
      setZoom((z) => Math.max(1.0, Math.round((z - 0.2) * 100) / 100));
    }
  };

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">
              Laâyoune City Map — Hôpital Moulay Hassan to Place du Méchouar
            </h3>
            <span
              className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded ${
                greenWaveActive
                  ? 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                  : 'bg-[#1E293B] text-[#94A3B8] border border-[#334155]'
              }`}
            >
              {greenWaveActive ? 'GREEN ZONE ACTIVE · 5 TURNS LOCKED' : 'ORGANIC CITY CIRCULATION'}
            </span>
          </div>
          <p className="text-xs text-[#94A3B8] mt-0.5">
            {greenWaveActive
              ? 'Multi-turn route from Hospital → Place du Méchouar is now a GREEN ZONE. Opposite & cross streets turned RED.'
              : 'Realistic Laâyoune streets & roundabouts with random traffic. Drag to pan or use zoom controls.'}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setCars(createRandomizedCars())}
            title="Randomize Car Traffic Positions & Speeds"
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium bg-[#1E293B] hover:bg-[#334155] text-[#E2E8F0] rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Shuffle className="w-3.5 h-3.5 text-[#38BDF8]" />
            Randomize Cars
          </button>

          <button
            type="button"
            onClick={onToggleManualGreenWave}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              greenWaveActive
                ? 'bg-[#EF4444] text-white hover:bg-[#DC2626]'
                : 'bg-[#10B981] text-[#0B0F17] hover:bg-[#34D399]'
            }`}
          >
            {greenWaveActive ? (
              <>
                <RotateCcw className="w-3.5 h-3.5" />
                Reset to Normal Traffic
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                Test Faint & Dispatch Ambulance
              </>
            )}
          </button>
        </div>
      </div>

      {/* Live Ambulance Telemetry & Turn-by-Turn Navigation Bar */}
      <div
        className={`px-4 py-2.5 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3 text-xs transition-colors ${
          greenWaveActive ? 'bg-[#064E3B]/40' : 'bg-[#0B0F17]'
        }`}
      >
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              greenWaveActive ? 'bg-[#10B981] animate-ping' : 'bg-[#64748B]'
            }`}
          />
          <Siren
            className={`w-4 h-4 ${
              greenWaveActive ? 'text-[#10B981]' : 'text-[#64748B]'
            }`}
          />
          <span className="font-semibold text-white">Ambulance GPS (AMB-LAA-01):</span>
          <span className={greenWaveActive ? 'text-[#34D399] font-medium' : 'text-[#94A3B8]'}>
            {greenWaveActive ? currentTurnStatus : 'Standby at Hôpital Moulay Hassan'}
          </span>
        </div>

        {greenWaveActive ? (
          <div className="flex items-center gap-3 font-mono tabular-nums text-[11px]">
            <span className="text-[#CBD5E1]">
              Speed: <strong className="text-[#10B981]">72 km/h</strong>
            </span>
            <span className="text-[#CBD5E1]">
              Dist: <strong className="text-white">{remainingKm} km</strong>
            </span>
            <span className="text-[#CBD5E1]">
              ETA: <strong className="text-[#10B981]">{etaSeconds}s</strong>
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-4 text-[11px] text-[#94A3B8]">
            <span>
              Active Traffic: <strong className="text-white">{cars.length} vehicles circulating</strong>
            </span>
            <span>
              Destination: <strong className="text-[#06B6D4]">Place du Méchouar (Circle)</strong>
            </span>
          </div>
        )}
      </div>

      {/* Bigger, Zoomable & Pannable Realistic Laâyoune City Map Canvas */}
      <div
        onWheel={handleWheelZoom}
        className="relative h-[475px] w-full bg-[#111927] overflow-hidden select-none"
      >
        {/* Top-Right Interactive Map Zoom & Focus Controls */}
        <div className="absolute top-3 right-3 z-20 flex flex-col gap-1.5 bg-[#0B0F17]/90 backdrop-blur-md border border-[#1E293B] p-1.5 rounded-xl shadow-lg">
          <button
            type="button"
            onClick={handleZoomIn}
            title="Zoom In"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-white transition-colors flex items-center justify-center"
          >
            <ZoomIn className="w-4 h-4 text-[#06B6D4]" />
          </button>
          <div className="text-[10px] font-mono tabular-nums text-center text-[#94A3B8] py-0.5">
            {zoom.toFixed(1)}x
          </div>
          <button
            type="button"
            onClick={handleZoomOut}
            title="Zoom Out"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-white transition-colors flex items-center justify-center"
          >
            <ZoomOut className="w-4 h-4 text-[#06B6D4]" />
          </button>
          <button
            type="button"
            onClick={handleResetView}
            title="Reset Map Zoom"
            className="p-2 rounded-lg bg-[#161F30] hover:bg-[#1E293B] text-[#CBD5E1] hover:text-white transition-colors flex items-center justify-center"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Top-Left Quick Focus Buttons (Zoom to Place du Méchouar / Follow Ambulance) */}
        <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={handleFocusPlaceMechouar}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-[#0B0F17]/90 hover:bg-[#161F30] text-[#06B6D4] border border-[#06B6D4]/40 backdrop-blur-md shadow"
          >
            <Crosshair className="w-3.5 h-3.5" />
            Zoom to Place du Méchouar
          </button>

          {greenWaveActive && (
            <button
              type="button"
              onClick={() => {
                setFollowAmbulance((prev) => !prev);
                if (!followAmbulance && zoom < 1.6) setZoom(1.75);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold border backdrop-blur-md shadow transition-colors ${
                followAmbulance
                  ? 'bg-[#10B981] text-[#0B0F17] border-[#10B981]'
                  : 'bg-[#0B0F17]/90 text-[#10B981] border-[#10B981]/40 hover:bg-[#161F30]'
              }`}
            >
              <Siren className="w-3.5 h-3.5" />
              {followAmbulance ? 'Following Ambulance' : 'Follow Ambulance'}
            </button>
          )}
        </div>

        <svg
          viewBox={`${viewBoxX} ${viewBoxY} ${Math.round(viewW)} ${Math.round(viewH)}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className={`w-full h-full object-cover ${
            isDragging ? 'cursor-grabbing' : 'cursor-grab'
          }`}
          aria-label="Laâyoune Organic Multi-Turn Traffic Map with Place du Méchouar Circle"
        >
          {/* Base Laâyoune Terrain Canvas */}
          <rect width="1080" height="620" fill="#111927" />

          {/* Natural Oued Sakia El Hamra Dry Riverbed / Green Belt in North-West Corner */}
          <path
            d="M 0,18 Q 140,35 280,15 T 560,12 L 560,0 L 0,0 Z"
            fill="#102826"
            opacity="0.7"
          />

          {/* Urban Parks & Green Zones */}
          <polygon
            points="615,52 745,42 758,108 625,155"
            fill="#112A28"
            stroke="#1B433E"
            strokeWidth="1.2"
          />
          <polygon
            points="45,405 115,402 102,468 38,465"
            fill="#112A28"
            stroke="#1B433E"
            strokeWidth="1.2"
          />

          {/* Irregular Realistic City Blocks */}
          <g fill="#182234" stroke="#23324A" strokeWidth="1.2" strokeLinejoin="round">
            {IRREGULAR_CITY_BLOCKS.map((pts, i) => (
              <polygon key={`blk-${i}`} points={pts} />
            ))}
          </g>

          {/* Secondary Organic Neighborhood Streets */}
          {SECONDARY_STREETS.map((st, idx) => (
            <polyline
              key={`sec-st-${idx}`}
              points={toSvgPoints(st)}
              fill="none"
              stroke="#202D42"
              strokeWidth="13"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}

          {/* ================= MAIN ORGANIC LAÂYOUNE AVENUES (ASPHALT) ================= */}
          {[
            CITY_ROUTES['av-hassan-nw-se'],
            CITY_ROUTES['bd-mekka-we'],
            CITY_ROUTES['av-massira-sn'],
            CITY_ROUTES['rue-smara-we'],
            CITY_ROUTES['bd-kairouan-ew'],
            CITY_ROUTES['av-mohammed-v'],
            CITY_ROUTES['av-sakia-north'],
          ].map((pts, idx) => (
            <g key={`main-av-${idx}`}>
              <polyline
                points={toSvgPoints(pts)}
                fill="none"
                stroke="#2C3C55"
                strokeWidth={idx === 1 ? '32' : '26'}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(pts)}
                fill="none"
                stroke="#475569"
                strokeWidth="1"
                strokeDasharray="7 7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          ))}

          {/* Roundabout Asphalt Rings at Turn 1 (Kairouan) & Turn 3 (Al Massira) */}
          {[
            { cx: 255, cy: 445, r: 22 },
            { cx: 515, cy: 325, r: 24 },
          ].map((rb, i) => (
            <g key={`rb-${i}`}>
              <circle
                cx={rb.cx}
                cy={rb.cy}
                r={rb.r}
                fill="#111927"
                stroke="#2C3C55"
                strokeWidth="14"
              />
              {/* Empty circle inside roundabout */}
              <circle
                cx={rb.cx}
                cy={rb.cy}
                r={rb.r - 8}
                fill="none"
                stroke="#38BDF8"
                strokeOpacity="0.35"
                strokeWidth="1.5"
              />
            </g>
          ))}

          {/* ===================================================================== */}
          {/* PLACE DU MÉCHOUAR — LARGE EMPTY CIRCLE PLAZA & ROUNDABOUT RING        */}
          {/* ===================================================================== */}
          <g
            transform={`translate(${PLACE_MECHOUAR_CENTER[0]}, ${PLACE_MECHOUAR_CENTER[1]})`}
          >
            {/* Outer Circular Roadway of Place du Méchouar */}
            <circle
              r={PLACE_MECHOUAR_RADIUS}
              fill="none"
              stroke="#2C3C55"
              strokeWidth="26"
            />
            {/* Dashed circular lane divider */}
            <circle
              r={PLACE_MECHOUAR_RADIUS}
              fill="none"
              stroke="#475569"
              strokeWidth="1.2"
              strokeDasharray="6 6"
            />

            {/* THE EMPTY CIRCLE OF PLACE DU MÉCHOUAR (Strictly empty interior as requested) */}
            <circle
              r={PLACE_MECHOUAR_RADIUS - 14}
              fill="#111927"
              stroke={greenWaveActive ? '#EF4444' : '#06B6D4'}
              strokeWidth={greenWaveActive ? '3.5' : '2.5'}
            />
            <circle
              r={PLACE_MECHOUAR_RADIUS - 22}
              fill="none"
              stroke={greenWaveActive ? '#EF4444' : '#06B6D4'}
              strokeOpacity="0.45"
              strokeWidth="1.2"
              strokeDasharray="5 4"
            />

            {/* Subtle label arched/centered for Place du Méchouar */}
            <text
              x="0"
              y="-4"
              textAnchor="middle"
              fill={greenWaveActive ? '#FCA5A5' : '#E2E8F0'}
              fontSize="10.5"
              fontWeight="800"
            >
              Place du Méchouar
            </text>
            <text
              x="0"
              y="10"
              textAnchor="middle"
              fill={greenWaveActive ? '#EF4444' : '#06B6D4'}
              fontSize="9"
              fontWeight="700"
            >
              ساحة المشور
            </text>
          </g>

          {/* ================= EMERGENCY FAINT OVERLAYS (HIDDEN UNTIL FAINT!) ================= */}
          {greenWaveActive && (
            <g>
              {/* 1. OPPOSITE & INTERSECTING STREETS TURN RED (CARS STOPPED) */}
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeOpacity="0.28"
                strokeWidth="14"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeWidth="4.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Intersecting Feeder Streets Locked RED */}
              {[
                // North of Turn 2 on Av. Hassan II
                [[355, 35], [330, 155], [305, 275]],
                // West of Turn 2 on Bd. Mekka
                [[35, 240], [175, 265], [290, 286]],
                // East of Turn 3 on Bd. Mekka
                [[535, 328], [715, 355], [1040, 405]],
                // South of Turn 3 on Av. Al Massira
                [[415, 585], [465, 455], [508, 342]],
                // North of Turn 4 on Av. Al Massira
                [[695, 35], [618, 172]],
                // West of Turn 4 on Rue Smara
                [[45, 115], [195, 140], [330, 155], [475, 168], [595, 182]],
                // South of Turn 5 on Av. Mohammed V
                [[895, 585], [820, 495], [715, 355], [730, 226]],
                // East of Turn 1 on Bd. Kairouan
                [[275, 446], [465, 455], [640, 475], [820, 495], [1035, 525]],
              ].map((seg, i) => (
                <g key={`red-seg-${i}`}>
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeOpacity="0.20"
                    strokeWidth="20"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeWidth="4.5"
                    strokeDasharray="10 6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              ))}

              {/* 2. MULTI-TURN HOSPITAL -> PLACE DU MÉCHOUAR GREEN ZONE */}
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#10B981"
                strokeOpacity="0.35"
                strokeWidth="24"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#10B981"
                strokeWidth="9"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(HOSPITAL_TO_MECHOUAR_PATH)}
                fill="none"
                stroke="#D1FAE5"
                strokeWidth="3"
                strokeDasharray="8 12"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Green Ring Highlight around Place du Méchouar Empty Circle */}
              <circle
                cx={PLACE_MECHOUAR_CENTER[0]}
                cy={PLACE_MECHOUAR_CENTER[1]}
                r={PLACE_MECHOUAR_RADIUS}
                fill="none"
                stroke="#10B981"
                strokeOpacity="0.45"
                strokeWidth="12"
              />
            </g>
          )}

          {/* ================= STREET NAMES ALONG ANGLED AVENUES ================= */}
          <text
            x="415"
            y="302"
            fill="#CBD5E1"
            fontSize="10.5"
            fontWeight="700"
            transform="rotate(9, 415, 302)"
          >
            Boulevard Mekka · شارع مكة
          </text>
          <text
            x="415"
            y="156"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(6, 415, 156)"
          >
            Rue Smara
          </text>
          <text
            x="175"
            y="462"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-19, 175, 462)"
          >
            Rue de l&apos;Hôpital
          </text>
          <text
            x="560"
            y="460"
            fill="#94A3B8"
            fontSize="10"
            fontWeight="600"
            transform="rotate(6, 560, 460)"
          >
            Bd. Kairouan
          </text>
          <text
            x="306"
            y="215"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-79, 306, 215)"
          >
            Av. Hassan II
          </text>
          <text
            x="560"
            y="245"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-55, 560, 245)"
          >
            Av. Al Massira
          </text>
          <text
            x="738"
            y="295"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-84, 738, 295)"
          >
            Av. Mohammed V
          </text>

          {/* ================= SIMULATED RANDOMIZED CITY CARS ================= */}
          {cars.map((car) => {
            const path = CITY_ROUTES[car.routeId] || HOSPITAL_TO_MECHOUAR_PATH;
            const pt = getPointOnPolyline(path, car.progress, car.laneOffset);
            const isStoppedRed = greenWaveActive && !car.isGreenCorridorDirection;

            return (
              <g
                key={car.id}
                transform={`translate(${pt.x}, ${pt.y}) rotate(${pt.angle}) scale(${car.sizeScale})`}
              >
                <rect
                  x="-8"
                  y="-4.5"
                  width="16"
                  height="9"
                  rx="2.5"
                  fill={isStoppedRed ? '#EF4444' : car.color}
                  stroke="#0B0F17"
                  strokeWidth="1.2"
                />
                <rect x="2" y="-3" width="3.5" height="6" rx="1" fill="#0F172A" />
                {isStoppedRed && (
                  <circle cx="-9" cy="0" r="3.5" fill="#EF4444" fillOpacity="0.85" />
                )}
              </g>
            );
          })}

          {/* ================= 5 MULTI-TURN INTERSECTION TRAFFIC SIGNALS ================= */}
          {CORRIDOR_TURN_INTERSECTIONS.map((node, index) => (
            <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
              <circle
                r="11"
                fill="#0B0F17"
                stroke={greenWaveActive ? '#10B981' : '#64748B'}
                strokeWidth="2"
              />
              <circle cx="-3.5" cy="0" r="3" fill="#10B981" />
              <circle
                cx="3.5"
                cy="0"
                r="3"
                fill={greenWaveActive ? '#EF4444' : '#F59E0B'}
              />
              {greenWaveActive && (
                <g transform="translate(0, 22)">
                  <rect
                    x="-46"
                    y="-8"
                    width="92"
                    height="14"
                    rx="3.5"
                    fill="#0B0F17"
                    stroke="#10B981"
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="2"
                    textAnchor="middle"
                    fill="#10B981"
                    fontSize="8.5"
                    fontWeight="700"
                  >
                    T{index + 1}: GREEN / RED
                  </text>
                </g>
              )}
            </g>
          ))}

          {/* ================= LIVE AMBULANCE TRACKER FOLLOWING ALL 5 TURNS ================= */}
          {greenWaveActive && (
            <g transform={`translate(${ambulanceCoord.x}, ${ambulanceCoord.y})`}>
              <circle r="28" fill="#10B981" fillOpacity="0.24" />
              <circle r="17" fill="#EF4444" fillOpacity="0.28" />

              <g transform={`rotate(${ambulanceCoord.angle})`}>
                <rect
                  x="-15"
                  y="-8"
                  width="30"
                  height="16"
                  rx="4"
                  fill="#FFFFFF"
                  stroke="#0B0F17"
                  strokeWidth="2"
                />
                <rect x="-4.5" y="-1.5" width="9" height="3" fill="#EF4444" />
                <rect x="-1.5" y="-4.5" width="3" height="9" fill="#EF4444" />
                <rect x="7" y="-5.5" width="3.5" height="4.5" fill="#EF4444" />
                <rect x="7" y="1" width="3.5" height="4.5" fill="#3B82F6" />
              </g>

              <g transform="translate(0, -28)">
                <rect
                  x="-66"
                  y="-13"
                  width="132"
                  height="20"
                  rx="5"
                  fill="#10B981"
                  stroke="#0B0F17"
                  strokeWidth="1.5"
                />
                <text
                  x="0"
                  y="0"
                  textAnchor="middle"
                  fill="#0B0F17"
                  fontSize="9.5"
                  fontWeight="800"
                >
                  AMBULANCE · {etaSeconds}s
                </text>
              </g>
            </g>
          )}

          {/* ================= HOSPITAL ORIGIN PIN (BOTTOM-LEFT) ================= */}
          <g transform={`translate(${HOSPITAL_ORIGIN[0]}, ${HOSPITAL_ORIGIN[1]})`}>
            <circle
              r="19"
              fill="#3B82F6"
              fillOpacity="0.22"
              stroke="#3B82F6"
              strokeWidth="2"
            />
            <circle r="10.5" fill="#3B82F6" stroke="#FFFFFF" strokeWidth="2" />
            <text x="0" y="3.5" textAnchor="middle" fill="#FFFFFF" fontSize="10" fontWeight="800">
              H
            </text>
            <g transform="translate(18, 29)">
              <rect
                x="-84"
                y="-11"
                width="168"
                height="20"
                rx="4"
                fill="#0B0F17"
                stroke="#3B82F6"
                strokeWidth="1.2"
              />
              <text
                x="0"
                y="2.5"
                textAnchor="middle"
                fill="#FFFFFF"
                fontSize="10"
                fontWeight="700"
              >
                Hôpital Moulay Hassan
              </text>
            </g>
          </g>

          {/* ================= PLACE DU MÉCHOUAR CALLOUT BADGE (ABOVE THE EMPTY CIRCLE) ================= */}
          <g
            transform={`translate(${PLACE_MECHOUAR_CENTER[0]}, ${
              PLACE_MECHOUAR_CENTER[1] - PLACE_MECHOUAR_RADIUS - 24
            })`}
          >
            <rect
              x="-104"
              y="-13"
              width="208"
              height="22"
              rx="5"
              fill={greenWaveActive ? '#EF4444' : '#0B0F17'}
              stroke={greenWaveActive ? '#FFFFFF' : '#06B6D4'}
              strokeWidth="1.4"
            />
            <text
              x="0"
              y="2"
              textAnchor="middle"
              fill="#FFFFFF"
              fontSize="10.5"
              fontWeight="800"
            >
              {greenWaveActive
                ? 'FAINT ALERT: Place du Méchouar'
                : 'Place du Méchouar (Test Loc)'}
            </text>
          </g>
        </svg>

        {/* Bottom Status Legend */}
        <div className="absolute bottom-2.5 left-3 right-3 bg-[#0B0F17]/90 backdrop-blur-md border border-[#1E293B] rounded-lg px-3 py-1.5 text-xs flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-4">
            {greenWaveActive ? (
              <>
                <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                  <ShieldCheck className="w-4 h-4" />
                  5-Turn Hospital → Place du Méchouar: GREEN ZONE
                </span>
                <span className="flex items-center gap-1.5 text-[#EF4444] font-semibold">
                  <OctagonAlert className="w-4 h-4" />
                  Opposite & Cross Routes: RED (Stopped)
                </span>
              </>
            ) : (
              <>
                <span className="flex items-center gap-1.5 text-[#CBD5E1]">
                  <Navigation className="w-3.5 h-3.5 text-[#38BDF8]" />
                  Organic City Traffic ({cars.length} Random Cars)
                </span>
                <span className="flex items-center gap-1.5 text-[#94A3B8]">
                  <MapPin className="w-3.5 h-3.5 text-[#06B6D4]" />
                  Place du Méchouar (Circular Plaza), Laâyoune
                </span>
              </>
            )}
          </div>
          <span className="text-[11px] font-mono text-[#94A3B8]">
            Scroll or click +/- to Zoom · Drag to Pan
          </span>
        </div>
      </div>

      {/* Camera Localization Bar */}
      <div className="p-3 bg-[#0B0F17] border-t border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs text-[#94A3B8]">Active Laâyoune Camera Localization:</span>
        <div className="flex flex-wrap items-center gap-1.5">
          {LAAYOUNE_CAMERA_NODES.map((cam) => (
            <button
              key={cam.id}
              type="button"
              onClick={() => onSelectCamera(cam)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeCameraId === cam.id
                  ? 'bg-[#06B6D4] text-[#0B0F17] font-semibold'
                  : 'bg-[#161F30] text-[#CBD5E1] hover:text-white'
              }`}
            >
              {cam.code}: {cam.name.split(' (')[0]}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

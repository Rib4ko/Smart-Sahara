import React, { useEffect, useState, useRef } from 'react';
import {
  Navigation,
  Siren,
  MapPin,
  ShieldCheck,
  OctagonAlert,
  Play,
  RotateCcw,
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
  speed: number;
  color: string;
}

// Interpolate [x, y, angleDeg] along any multi-turn polyline path
function getPointOnPolyline(
  points: [number, number][],
  t: number
): { x: number; y: number; angle: number } {
  const clamped = Math.max(0, Math.min(1, t));
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
      const x = points[i][0] + (points[i + 1][0] - points[i][0]) * localT;
      const y = points[i][1] + (points[i + 1][1] - points[i][1]) * localT;
      const angle =
        (Math.atan2(points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0]) * 180) /
        Math.PI;
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

// ============================================================================
// REALISTIC SMALL-CITY ROAD NETWORK OF LAÂYOUNE (1000 x 540 ViewBox)
// ============================================================================

// 1. MULTI-TURN EMERGENCY ROUTE (Hospital -> 5 City Turns -> School)
// Starts at Hôpital Moulay Hassan (110, 445) -> East on Rue de l'Hôpital ->
// Turn 1 North on Av. Hassan II -> Turn 2 East on Boulevard Mekka ->
// Turn 3 North on Av. Al Massira -> Turn 4 East on Rue Smara ->
// Turn 5 North into Lycée Hassan II School (810, 75)
const HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH: [number, number][] = [
  [110, 445], // Hôpital Moulay Hassan Gate
  [275, 445], // Turn 1: Corner of Rue de l'Hôpital & Av. Hassan II
  [275, 295], // Turn 2: Corner of Av. Hassan II & Boulevard Mekka
  [565, 295], // Turn 3: Corner of Boulevard Mekka & Av. Al Massira
  [565, 165], // Turn 4: Corner of Av. Al Massira & Rue Smara
  [810, 165], // Turn 5: Corner of Rue Smara & School Avenue
  [810, 75],  // Arrived: Lycée Hassan II School
];

// 2. OPPOSITE LANE ALONG THE SAME MULTI-TURN CORRIDOR (School -> Hospital, turns RED & STOPS on faint)
const OPPOSITE_CORRIDOR_PATH: [number, number][] = [
  [796, 75],
  [796, 151],
  [551, 151],
  [551, 281],
  [261, 281],
  [261, 431],
  [110, 431],
];

// 3. ADDITIONAL SMALL-CITY ROUTES (Cross streets, avenues & loops that turn RED where they intersect)
const CITY_ROUTES: Record<string, [number, number][]> = {
  // Primary multi-turn forward route
  'green-corridor': HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH,
  // Direct opposite lane
  'opp-corridor': OPPOSITE_CORRIDOR_PATH,
  // Full Avenue Hassan II (North -> South across Turn 1 & Turn 2)
  'av-hassan-ns': [
    [268, 25],
    [268, 160],
    [268, 290],
    [268, 440],
    [268, 515],
  ],
  // Full Boulevard Mekka West-to-East continuation (East -> West across Turn 2 & Turn 3)
  'bd-mekka-ew': [
    [965, 288],
    [810, 288],
    [560, 288],
    [270, 288],
    [35, 288],
  ],
  // Full Avenue Al Massira (North -> South across Turn 3 & Turn 4)
  'av-massira-ns': [
    [558, 25],
    [558, 160],
    [558, 290],
    [558, 440],
    [558, 515],
  ],
  // Rue Smara Westward (West -> East across Av. Hassan II & Av. Al Massira)
  'rue-smara-we': [
    [35, 158],
    [270, 158],
    [560, 158],
    [810, 158],
    [965, 158],
  ],
  // Boulevard Kairouan Southern Arterial (East -> West across southern city)
  'bd-kairouan-ew': [
    [965, 438],
    [810, 438],
    [560, 438],
    [270, 438],
    [110, 438],
  ],
  // School Avenue North-South (South -> North from Bd. Kairouan to School)
  'av-school-sn': [
    [803, 515],
    [803, 440],
    [803, 290],
    [803, 160],
    [803, 75],
  ],
};

// Background secondary neighborhood streets to give a true small-city grid look
const SECONDARY_STREETS: [number, number][][] = [
  [[135, 25], [135, 515]],
  [[415, 25], [415, 515]],
  [[685, 25], [685, 515]],
  [[915, 25], [915, 515]],
  [[35, 82], [965, 82]],
  [[35, 222], [965, 222]],
  [[35, 365], [965, 365]],
];

const INITIAL_CARS: SimCar[] = [
  // Cars on the forward corridor (speed up & clear path on Green Wave)
  { id: 'c-fwd-1', routeId: 'green-corridor', isGreenCorridorDirection: true, progress: 0.22, speed: 0.0024, color: '#38BDF8' },
  { id: 'c-fwd-2', routeId: 'green-corridor', isGreenCorridorDirection: true, progress: 0.64, speed: 0.0026, color: '#E2E8F0' },

  // Cars on the opposite lane of the multi-turn route (STOP on faint when route turns RED)
  { id: 'c-opp-1', routeId: 'opp-corridor', isGreenCorridorDirection: false, progress: 0.16, speed: 0.0022, color: '#FBBF24' },
  { id: 'c-opp-2', routeId: 'opp-corridor', isGreenCorridorDirection: false, progress: 0.48, speed: 0.0024, color: '#F8FAFC' },
  { id: 'c-opp-3', routeId: 'opp-corridor', isGreenCorridorDirection: false, progress: 0.79, speed: 0.0021, color: '#94A3B8' },

  // Cars on intersecting city avenues (STOP on faint when opposite/cross routes turn RED)
  { id: 'c-has-1', routeId: 'av-hassan-ns', isGreenCorridorDirection: false, progress: 0.14, speed: 0.0025, color: '#CBD5E1' },
  { id: 'c-has-2', routeId: 'av-hassan-ns', isGreenCorridorDirection: false, progress: 0.72, speed: 0.0023, color: '#FBBF24' },
  { id: 'c-mek-1', routeId: 'bd-mekka-ew', isGreenCorridorDirection: false, progress: 0.19, speed: 0.0022, color: '#38BDF8' },
  { id: 'c-mek-2', routeId: 'bd-mekka-ew', isGreenCorridorDirection: false, progress: 0.68, speed: 0.0025, color: '#E2E8F0' },
  { id: 'c-mas-1', routeId: 'av-massira-ns', isGreenCorridorDirection: false, progress: 0.18, speed: 0.0026, color: '#F8FAFC' },
  { id: 'c-mas-2', routeId: 'av-massira-ns', isGreenCorridorDirection: false, progress: 0.77, speed: 0.0022, color: '#94A3B8' },
  { id: 'c-sma-1', routeId: 'rue-smara-we', isGreenCorridorDirection: false, progress: 0.24, speed: 0.0023, color: '#FBBF24' },
  { id: 'c-kai-1', routeId: 'bd-kairouan-ew', isGreenCorridorDirection: false, progress: 0.35, speed: 0.0024, color: '#CBD5E1' },
  { id: 'c-sch-1', routeId: 'av-school-sn', isGreenCorridorDirection: false, progress: 0.29, speed: 0.0025, color: '#38BDF8' },
];

// The 5 Turn Intersections along the Hospital -> School Route
const CORRIDOR_TURN_INTERSECTIONS = [
  { id: 'turn-1', label: 'Turn 1: Av. Hassan II', x: 275, y: 445 },
  { id: 'turn-2', label: 'Turn 2: Bd. Mekka', x: 275, y: 295 },
  { id: 'turn-3', label: 'Turn 3: Av. Al Massira', x: 565, y: 295 },
  { id: 'turn-4', label: 'Turn 4: Rue Smara', x: 565, y: 165 },
  { id: 'turn-5', label: 'Turn 5: School Gate', x: 810, y: 165 },
];

export const LaayouneCorridorMap: React.FC<LaayouneCorridorMapProps> = ({
  greenWaveActive,
  activeCameraId,
  onSelectCamera,
  onToggleManualGreenWave,
}) => {
  const [cars, setCars] = useState<SimCar[]>(INITIAL_CARS);
  const [ambulanceProgress, setAmbulanceProgress] = useState<number>(0);
  const greenWaveRef = useRef<boolean>(greenWaveActive);

  useEffect(() => {
    greenWaveRef.current = greenWaveActive;
    if (!greenWaveActive) {
      setAmbulanceProgress(0);
    }
  }, [greenWaveActive]);

  // 30 FPS City Traffic & Multi-Turn Ambulance Tracker Loop
  useEffect(() => {
    const interval = setInterval(() => {
      const isEmergency = greenWaveRef.current;

      setCars((prevCars) =>
        prevCars.map((car) => {
          // When person faints, all cars on opposite & cross routes STOP at the red lights
          if (isEmergency && !car.isGreenCorridorDirection) {
            return car;
          }

          const boost = isEmergency && car.isGreenCorridorDirection ? 1.55 : 1.0;
          let nextProg = car.progress + car.speed * boost;
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
          return Math.min(1, prev + 0.0034);
        });
      }
    }, 35);

    return () => clearInterval(interval);
  }, []);

  const ambulanceCoord = getPointOnPolyline(
    HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH,
    ambulanceProgress
  );
  const remainingKm = Math.max(0, (1 - ambulanceProgress) * 3.1).toFixed(2);
  const etaSeconds = Math.max(0, Math.ceil((1 - ambulanceProgress) * 115));

  const currentTurnStatus =
    ambulanceProgress < 0.16
      ? "Rue de l'Hôpital → Approaching Turn 1"
      : ambulanceProgress < 0.34
      ? 'Turn 1 Left → North on Av. Hassan II'
      : ambulanceProgress < 0.58
      ? 'Turn 2 Right → East on Boulevard Mekka'
      : ambulanceProgress < 0.73
      ? 'Turn 3 Left → North on Av. Al Massira'
      : ambulanceProgress < 0.91
      ? 'Turn 4 Right → East on Rue Smara'
      : ambulanceProgress < 0.98
      ? 'Turn 5 Left → School Gate Approach'
      : 'Arrived at Lycée Hassan II School';

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">
              Laâyoune City Map — Multi-Turn Hospital to School Corridor
            </h3>
            <span
              className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded ${
                greenWaveActive
                  ? 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                  : 'bg-[#1E293B] text-[#94A3B8] border border-[#334155]'
              }`}
            >
              {greenWaveActive ? 'GREEN ZONE ACTIVE · 5 TURNS LOCKED' : 'NORMAL CITY CIRCULATION'}
            </span>
          </div>
          <p className="text-xs text-[#94A3B8] mt-0.5">
            {greenWaveActive
              ? 'Multi-turn route from Hospital → School is now a GREEN ZONE. Opposite & cross streets turned RED and stopped.'
              : 'Normal car circulation across Laâyoune streets. Emergency route stays hidden until a faint is detected.'}
          </p>
        </div>

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
          <div className="flex items-center gap-3 font-mono text-[11px]">
            <span className="text-[#CBD5E1]">
              Speed: <strong className="text-[#10B981]">68 km/h</strong>
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
            <span>Active Traffic: <strong className="text-white">14 vehicles circulating</strong></span>
            <span>Route Overlay: <strong className="text-[#94A3B8]">Hidden until faint</strong></span>
          </div>
        )}
      </div>

      {/* Realistic Small-City Laâyoune Map Canvas */}
      <div className="relative h-[350px] w-full bg-[#141D2B] overflow-hidden select-none">
        <svg
          viewBox="0 0 1000 540"
          className="w-full h-full object-cover"
          aria-label="Laâyoune Small City Multi-Turn Traffic Map"
        >
          {/* Base Map Canvas */}
          <rect width="1000" height="540" fill="#131B28" />

          {/* Urban City Blocks & Neighborhood Buildings */}
          <g fill="#1B2638" stroke="#243248" strokeWidth="1">
            {/* Row 1 Blocks */}
            <rect x="45" y="25" width="75" height="45" rx="4" />
            <rect x="150" y="25" width="95" height="110" rx="5" />
            <rect x="295" y="25" width="105" height="110" rx="5" />
            <rect x="430" y="25" width="105" height="110" rx="5" />
            <rect x="585" y="25" width="85" height="110" rx="5" />
            <rect x="700" y="25" width="80" height="110" rx="5" />
            <rect x="835" y="25" width="115" height="110" rx="5" />

            {/* Row 2 Blocks */}
            <rect x="45" y="180" width="75" height="90" rx="5" />
            <rect x="150" y="180" width="95" height="90" rx="5" />
            <rect x="295" y="180" width="105" height="90" rx="5" />
            <rect x="585" y="180" width="85" height="90" rx="5" />
            <rect x="700" y="180" width="80" height="90" rx="5" />
            <rect x="835" y="180" width="115" height="90" rx="5" />

            {/* Row 3 Blocks */}
            <rect x="45" y="315" width="75" height="100" rx="5" />
            <rect x="150" y="315" width="95" height="100" rx="5" />
            <rect x="295" y="315" width="105" height="100" rx="5" />
            <rect x="430" y="315" width="105" height="100" rx="5" />
            <rect x="585" y="315" width="85" height="100" rx="5" />
            <rect x="700" y="315" width="80" height="100" rx="5" />
            <rect x="835" y="315" width="115" height="100" rx="5" />

            {/* Row 4 Southern Blocks */}
            <rect x="150" y="465" width="95" height="50" rx="4" />
            <rect x="295" y="465" width="240" height="50" rx="4" />
            <rect x="585" y="465" width="195" height="50" rx="4" />
            <rect x="835" y="465" width="115" height="50" rx="4" />
          </g>

          {/* Place El Méchouar Green Plaza in Central Laâyoune */}
          <rect
            x="430"
            y="180"
            width="105"
            height="90"
            rx="6"
            fill="#12302B"
            stroke="#1E4B43"
            strokeWidth="1.5"
          />
          <text
            x="482"
            y="228"
            textAnchor="middle"
            fill="#34D399"
            fillOpacity="0.7"
            fontSize="10"
            fontWeight="600"
          >
            Place El Méchouar
          </text>

          {/* Secondary Neighborhood Streets */}
          {SECONDARY_STREETS.map((st, idx) => (
            <polyline
              key={`sec-st-${idx}`}
              points={toSvgPoints(st)}
              fill="none"
              stroke="#222F43"
              strokeWidth="12"
              strokeLinecap="round"
            />
          ))}

          {/* ================= MAIN LAÂYOUNE ARTERIAL AVENUES (NORMAL ASPHALT) ================= */}
          {/* Horizontal Avenues */}
          {/* Rue Smara (Y = 160) */}
          <polyline
            points="35,160 965,160"
            fill="none"
            stroke="#2D3D54"
            strokeWidth="28"
            strokeLinecap="round"
          />
          {/* Boulevard Mekka (Y = 290) */}
          <polyline
            points="35,290 965,290"
            fill="none"
            stroke="#334155"
            strokeWidth="34"
            strokeLinecap="round"
          />
          {/* Rue de l'Hôpital / Boulevard Kairouan (Y = 440) */}
          <polyline
            points="35,440 965,440"
            fill="none"
            stroke="#2D3D54"
            strokeWidth="28"
            strokeLinecap="round"
          />

          {/* Vertical Avenues */}
          {/* Avenue Hassan II (X = 270) */}
          <polyline
            points="270,25 270,515"
            fill="none"
            stroke="#2D3D54"
            strokeWidth="28"
            strokeLinecap="round"
          />
          {/* Avenue Al Massira (X = 560) */}
          <polyline
            points="560,25 560,515"
            fill="none"
            stroke="#2D3D54"
            strokeWidth="28"
            strokeLinecap="round"
          />
          {/* Avenue de l'École / Lycée Hassan II Approach (X = 805) */}
          <polyline
            points="805,25 805,515"
            fill="none"
            stroke="#2D3D54"
            strokeWidth="28"
            strokeLinecap="round"
          />

          {/* Subtle Center Dashed Lane Dividers on Main Avenues */}
          <g stroke="#475569" strokeWidth="1" strokeDasharray="6 6" fill="none">
            <line x1="35" y1="160" x2="965" y2="160" />
            <line x1="35" y1="290" x2="965" y2="290" />
            <line x1="35" y1="440" x2="965" y2="440" />
            <line x1="270" y1="25" x2="270" y2="515" />
            <line x1="560" y1="25" x2="560" y2="515" />
            <line x1="805" y1="25" x2="805" y2="515" />
          </g>

          {/* ================= EMERGENCY FAINT OVERLAYS (HIDDEN UNTIL FAINT!) ================= */}
          {greenWaveActive && (
            <g>
              {/* 1. OPPOSITE & INTERSECTING STREETS TURN RED (CARS STOPPED) */}
              {/* Opposite lane along the multi-turn route turns RED */}
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeOpacity="0.3"
                strokeWidth="14"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={toSvgPoints(OPPOSITE_CORRIDOR_PATH)}
                fill="none"
                stroke="#EF4444"
                strokeWidth="5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Cross-streets feeding into Turn 1, Turn 2, Turn 3, Turn 4, Turn 5 locked RED */}
              {[
                // North of Turn 2 on Av. Hassan II
                [[268, 25], [268, 285]],
                // West of Turn 2 on Bd. Mekka
                [[35, 288], [265, 288]],
                // East of Turn 3 on Bd. Mekka
                [[575, 288], [965, 288]],
                // South of Turn 3 on Av. Al Massira
                [[558, 305], [558, 515]],
                // North of Turn 4 on Av. Al Massira
                [[558, 25], [558, 155]],
                // West of Turn 4 on Rue Smara
                [[35, 158], [550, 158]],
                // East of Turn 5 on Rue Smara
                [[820, 158], [965, 158]],
                // South of Turn 5 on School Avenue
                [[803, 175], [803, 515]],
                // East of Turn 1 on Bd. Kairouan
                [[285, 438], [965, 438]],
              ].map((seg, i) => (
                <g key={`red-seg-${i}`}>
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeOpacity="0.22"
                    strokeWidth="22"
                    strokeLinecap="round"
                  />
                  <polyline
                    points={toSvgPoints(seg as [number, number][])}
                    fill="none"
                    stroke="#EF4444"
                    strokeWidth="5"
                    strokeDasharray="10 6"
                    strokeLinecap="round"
                  />
                </g>
              ))}

              {/* 2. MULTI-TURN HOSPITAL -> SCHOOL GREEN ZONE */}
              {/* Wide glowing green corridor zone */}
              <polyline
                points={toSvgPoints(HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH)}
                fill="none"
                stroke="#10B981"
                strokeOpacity="0.35"
                strokeWidth="24"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {/* Crisp solid green route line */}
              <polyline
                points={toSvgPoints(HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH)}
                fill="none"
                stroke="#10B981"
                strokeWidth="9"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {/* Inner bright directional dashes showing the multi-turn flow */}
              <polyline
                points={toSvgPoints(HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH)}
                fill="none"
                stroke="#D1FAE5"
                strokeWidth="3"
                strokeDasharray="8 12"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          )}

          {/* ================= STREET NAMES ================= */}
          <text x="420" y="282" fill="#CBD5E1" fontSize="10" fontWeight="700">
            Boulevard Mekka · شارع مكة
          </text>
          <text x="680" y="152" fill="#CBD5E1" fontSize="10" fontWeight="600">
            Rue Smara
          </text>
          <text x="165" y="432" fill="#CBD5E1" fontSize="10" fontWeight="600">
            Rue de l&apos;Hôpital
          </text>
          <text x="650" y="432" fill="#94A3B8" fontSize="10" fontWeight="600">
            Bd. Kairouan
          </text>
          <text
            x="258"
            y="225"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-90, 258, 225)"
          >
            Av. Hassan II
          </text>
          <text
            x="548"
            y="235"
            fill="#CBD5E1"
            fontSize="10"
            fontWeight="600"
            transform="rotate(-90, 548, 235)"
          >
            Av. Al Massira
          </text>

          {/* ================= SIMULATED CITY CARS ================= */}
          {cars.map((car) => {
            const path = CITY_ROUTES[car.routeId] || HOSPITAL_TO_SCHOOL_MULTI_TURN_PATH;
            const pt = getPointOnPolyline(path, car.progress);
            const isStoppedRed = greenWaveActive && !car.isGreenCorridorDirection;

            return (
              <g
                key={car.id}
                transform={`translate(${pt.x}, ${pt.y}) rotate(${pt.angle})`}
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
              {/* Green Corridor Light */}
              <circle cx="-3.5" cy="0" r="3" fill="#10B981" />
              {/* Opposite/Cross Traffic Light (Turns RED on faint) */}
              <circle
                cx="3.5"
                cy="0"
                r="3"
                fill={greenWaveActive ? '#EF4444' : '#F59E0B'}
              />
              {greenWaveActive && (
                <g transform="translate(0, 21)">
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
              <circle r="26" fill="#10B981" fillOpacity="0.24" />
              <circle r="16" fill="#EF4444" fillOpacity="0.28" />

              <g transform={`rotate(${ambulanceCoord.angle})`}>
                <rect
                  x="-14"
                  y="-7.5"
                  width="28"
                  height="15"
                  rx="4"
                  fill="#FFFFFF"
                  stroke="#0B0F17"
                  strokeWidth="2"
                />
                {/* Red Medical Cross */}
                <rect x="-4.5" y="-1.5" width="9" height="3" fill="#EF4444" />
                <rect x="-1.5" y="-4.5" width="3" height="9" fill="#EF4444" />
                {/* Flashing Roof Siren */}
                <rect x="7" y="-5.5" width="3" height="4.5" fill="#EF4444" />
                <rect x="7" y="1" width="3" height="4.5" fill="#3B82F6" />
              </g>

              <g transform="translate(0, -26)">
                <rect
                  x="-64"
                  y="-13"
                  width="128"
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
          <g transform="translate(110, 445)">
            <circle
              r="18"
              fill="#3B82F6"
              fillOpacity="0.22"
              stroke="#3B82F6"
              strokeWidth="2"
            />
            <circle r="10" fill="#3B82F6" stroke="#FFFFFF" strokeWidth="2" />
            <text x="0" y="3.5" textAnchor="middle" fill="#FFFFFF" fontSize="10" fontWeight="800">
              H
            </text>
            <g transform="translate(15, 28)">
              <rect
                x="-82"
                y="-11"
                width="164"
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

          {/* ================= SCHOOL DESTINATION PIN (TOP-RIGHT) ================= */}
          <g transform="translate(810, 75)">
            <circle
              r={greenWaveActive ? 24 : 18}
              fill={greenWaveActive ? '#EF4444' : '#06B6D4'}
              fillOpacity="0.25"
              stroke={greenWaveActive ? '#EF4444' : '#06B6D4'}
              strokeWidth="2"
            />
            <circle
              r="10"
              fill={greenWaveActive ? '#EF4444' : '#06B6D4'}
              stroke="#FFFFFF"
              strokeWidth="2"
            />
            <text x="0" y="3.5" textAnchor="middle" fill="#FFFFFF" fontSize="10" fontWeight="800">
              S
            </text>
            <g transform="translate(0, -24)">
              <rect
                x="-92"
                y="-12"
                width="184"
                height="21"
                rx="4"
                fill={greenWaveActive ? '#EF4444' : '#0B0F17'}
                stroke={greenWaveActive ? '#FFFFFF' : '#06B6D4'}
                strokeWidth="1.2"
              />
              <text
                x="0"
                y="2"
                textAnchor="middle"
                fill="#FFFFFF"
                fontSize="10"
                fontWeight="700"
              >
                {greenWaveActive
                  ? 'FAINT ALERT: Lycée Hassan II School'
                  : 'Lycée Hassan II School (Test Loc)'}
              </text>
            </g>
          </g>
        </svg>

        {/* Bottom Status Legend */}
        <div className="absolute bottom-2.5 left-3 right-3 bg-[#0B0F17]/90 backdrop-blur-md border border-[#1E293B] rounded-lg px-3 py-1.5 text-xs flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-4">
            {greenWaveActive ? (
              <>
                <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                  <ShieldCheck className="w-4 h-4" />
                  5-Turn Hospital → School: GREEN ZONE
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
                  Normal Small-City Car Circulation (8 Routes)
                </span>
                <span className="flex items-center gap-1.5 text-[#94A3B8]">
                  <MapPin className="w-3.5 h-3.5 text-[#06B6D4]" />
                  Lycée Hassan II School, Laâyoune
                </span>
              </>
            )}
          </div>
          <span className="text-[11px] font-mono text-[#94A3B8]">
            Laâyoune City Grid · 5 Intersections
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

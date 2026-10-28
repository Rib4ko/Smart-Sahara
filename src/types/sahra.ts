export type OperationalMode = 'najda' | 'tifawt' | 'security';

export type TargetClothingColor = 'RED' | 'BLUE' | 'YELLOW' | 'BLACK' | 'WHITE' | 'GREEN';

export type VideoInputSource = 'webcam' | 'upload' | 'simulation';

export interface PoseKeypoint {
  name: string;
  x: number;
  y: number;
  confidence: number;
}

export interface DetectedSubject {
  id: number;
  label: string;
  bbox: [number, number, number, number];
  lastStandingBbox?: [number, number, number, number];
  headRoi: [number, number, number, number];
  torsoRoi: [number, number, number, number];
  keypoints: PoseKeypoint[];
  torsoColor: TargetClothingColor;
  hsvMatchPct: number;
  torsoAngleDeg: number;
  headDropPct: number;
  heightCompressionPct: number;
  aspectRatio: number;
  isFallen: boolean;
  isMemoryLocked: boolean;
  faintDurationSec: number;
  isConcealedFace: boolean;
  hasWeapon: boolean;
  weaponLabel?: string;
}

export interface TrafficSignalNode {
  id: string;
  name: string;
  arabicName: string;
  coordinates: string;
  xPct: number;
  yPct: number;
  etaSeconds: number;
}

export interface CameraSectorNode {
  id: string;
  code: string;
  name: string;
  sector: string;
  coordinates: string;
  xPct: number;
  yPct: number;
  defaultMode: OperationalMode;
}

export interface IncidentNotification {
  id: string;
  timestamp: string;
  mode: OperationalMode;
  title: string;
  description: string;
  cameraName: string;
  channel: 'WhatsApp' | 'Telegram';
  greenWaveTriggered: boolean;
}

export const TARGET_COLOR_CONFIG: Record<
  TargetClothingColor,
  {
    label: string;
    swatchHex: string;
  }
> = {
  RED: { label: 'Red', swatchHex: '#EF4444' },
  BLUE: { label: 'Blue / Teal', swatchHex: '#06B6D4' },
  YELLOW: { label: 'Yellow', swatchHex: '#EAB308' },
  GREEN: { label: 'Green', swatchHex: '#10B981' },
  BLACK: { label: 'Black', swatchHex: '#1E293B' },
  WHITE: { label: 'White', swatchHex: '#F8FAFC' },
};

export const LAAYOUNE_CAMERA_NODES: CameraSectorNode[] = [
  {
    id: 'cam-school-01',
    code: 'CAM-01',
    name: 'Lycée Hassan II School (Test Localization)',
    sector: 'School Zone · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 48,
    yPct: 46,
    defaultMode: 'najda',
  },
  {
    id: 'cam-mekka-02',
    code: 'CAM-02',
    name: 'Boulevard Mekka Crossing',
    sector: 'Main Avenue · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 33,
    yPct: 58,
    defaultMode: 'tifawt',
  },
  {
    id: 'cam-onee-03',
    code: 'CAM-03',
    name: 'ONEE Water Plant Gate',
    sector: 'Restricted Zone · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 19,
    yPct: 28,
    defaultMode: 'security',
  },
  {
    id: 'cam-est-04',
    code: 'CAM-04',
    name: 'EST Laâyoune Campus Entrance',
    sector: 'Campus Zone · Laâyoune',
    coordinates: 'Laâyoune, Morocco',
    xPct: 74,
    yPct: 67,
    defaultMode: 'security',
  },
];

export const BOULEVARD_MEKKA_SIGNALS: TrafficSignalNode[] = [
  {
    id: 'sig-01',
    name: 'Protection Civile Dispatch Base',
    arabicName: 'الوقاية المدنية',
    coordinates: '27.1481° N, 13.1942° W',
    xPct: 24,
    yPct: 68,
    etaSeconds: 18,
  },
  {
    id: 'sig-02',
    name: 'Place El Méchouar Junction',
    arabicName: 'مدار ساحة المشور',
    coordinates: '27.1505° N, 13.1988° W',
    xPct: 36,
    yPct: 57,
    etaSeconds: 42,
  },
  {
    id: 'sig-03',
    name: 'Boulevard Mekka × Av. Hassan II',
    arabicName: 'شارع مكة × الحسن الثاني',
    coordinates: '27.1536° N, 13.2033° W',
    xPct: 49,
    yPct: 46,
    etaSeconds: 75,
  },
  {
    id: 'sig-04',
    name: 'Boulevard Mekka North Circle',
    arabicName: 'مدار شارع مكة الشمالي',
    coordinates: '27.1569° N, 13.2075° W',
    xPct: 63,
    yPct: 36,
    etaSeconds: 105,
  },
  {
    id: 'sig-05',
    name: 'Moulay Hassan Hospital Emergency',
    arabicName: 'مستشفى مولاي الحسن المهدي',
    coordinates: '27.1604° N, 13.2118° W',
    xPct: 78,
    yPct: 25,
    etaSeconds: 138,
  },
];

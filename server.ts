import express from 'express';
import { createServer as createViteServer } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface IncidentRecord {
  id: string;
  timestamp: string;
  mode: 'najda' | 'tifawt' | 'security';
  title: string;
  description: string;
  cameraName: string;
  channel: 'WhatsApp' | 'Telegram';
  greenWaveTriggered: boolean;
}

// Server-side state for Laâyoune Smart City Backend
const serverState = {
  mode: 'najda' as 'najda' | 'tifawt' | 'security',
  targetColor: 'RED',
  greenWaveActive: false,
  greenWaveActivatedAt: null as number | null,
  activeCameraId: 'cam-mekka-01',
  incidents: [
    {
      id: 'inc-101',
      timestamp: '14:22:09',
      mode: 'najda',
      title: 'Medical Emergency — Person Collapsed',
      description: 'Ambulance dispatched to Boulevard Mekka. All 5 traffic lights switched to Green Wave.',
      cameraName: 'Boulevard Mekka Crossing',
      channel: 'WhatsApp',
      greenWaveTriggered: true,
    },
    {
      id: 'inc-102',
      timestamp: '14:18:41',
      mode: 'tifawt',
      title: 'Missing Person Match — Red Shirt (89%)',
      description: 'Located person wearing Red shirt at Place El Méchouar. Patrol notified.',
      cameraName: 'Place El Méchouar Plaza',
      channel: 'Telegram',
      greenWaveTriggered: false,
    },
  ] as IncidentRecord[],
};

const lastWebhookByMode: Record<string, number> = {};

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // GET /api/state — Returns current backend state & incident log
  app.get('/api/state', (_req, res) => {
    res.json(serverState);
  });

  // POST /api/mode — Updates active operational mode & target shirt color
  app.post('/api/mode', (req, res) => {
    const { mode, targetColor, activeCameraId } = req.body || {};
    if (mode && ['najda', 'tifawt', 'security'].includes(mode)) {
      serverState.mode = mode;
    }
    if (targetColor) {
      serverState.targetColor = String(targetColor).toUpperCase();
    }
    if (activeCameraId) {
      serverState.activeCameraId = activeCameraId;
    }
    res.json(serverState);
  });

  // POST /api/corridor — Toggles Boulevard Mekka Green-Wave Traffic Corridor
  app.post('/api/corridor', (req, res) => {
    const { active } = req.body || {};
    serverState.greenWaveActive = typeof active === 'boolean' ? active : !serverState.greenWaveActive;
    serverState.greenWaveActivatedAt = serverState.greenWaveActive ? Date.now() : null;
    res.json({
      greenWaveActive: serverState.greenWaveActive,
      activatedAt: serverState.greenWaveActivatedAt,
    });
  });

  // POST /api/dispatch — Records and dispatches an emergency alert with cooldown debouncing
  app.post('/api/dispatch', (req, res) => {
    const {
      mode = serverState.mode,
      title = 'Smart City Alert',
      description = 'Automated incident detected.',
      cameraName = 'Boulevard Mekka Crossing',
      channel = 'WhatsApp',
      force = false,
    } = req.body || {};

    const now = Date.now();
    // 6-second server-side debounce for automatic vision triggers unless manually forced
    if (!force && now - (lastWebhookByMode[mode] || 0) < 6000) {
      res.json({ skipped: true, reason: 'Cooldown active', incidents: serverState.incidents });
      return;
    }
    lastWebhookByMode[mode] = now;

    if (mode === 'najda') {
      serverState.greenWaveActive = true;
      serverState.greenWaveActivatedAt = now;
    }

    const record: IncidentRecord = {
      id: `inc-${now}`,
      timestamp: new Date().toTimeString().slice(0, 8),
      mode,
      title,
      description,
      cameraName,
      channel,
      greenWaveTriggered: mode === 'najda',
    };

    serverState.incidents = [record, ...serverState.incidents.slice(0, 19)];
    res.json({ skipped: false, incident: record, state: serverState });
  });

  // POST /api/reset — Clears active alert state and resets traffic lights to normal
  app.post('/api/reset', (_req, res) => {
    serverState.greenWaveActive = false;
    serverState.greenWaveActivatedAt = null;
    res.json(serverState);
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    if (fs.existsSync(distPath)) {
      app.use(express.static(distPath));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(distPath, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Sahra-Eye Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();

import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

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
  activeCameraId: 'cam-school-01',
  incidents: [
    {
      id: 'inc-101',
      timestamp: '14:22:09',
      mode: 'najda',
      title: 'Medical Emergency — Person Fainted at Place du Méchouar',
      description:
        'Hospital → Place du Méchouar route turned GREEN. Opposite cross-traffic routes locked RED.',
      cameraName: 'Place du Méchouar (Test Localization)',
      channel: 'WhatsApp',
      greenWaveTriggered: true,
    },
    {
      id: 'inc-102',
      timestamp: '14:18:41',
      mode: 'tifawt',
      title: 'Missing Person Match — Red Shirt (89%)',
      description: 'Located person wearing Red shirt at Boulevard Mekka Crossing. Patrol notified.',
      cameraName: 'Boulevard Mekka Crossing',
      channel: 'Telegram',
      greenWaveTriggered: false,
    },
  ] as IncidentRecord[],
};

const lastWebhookByMode: Record<string, number> = {};

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Support base64 frame payloads up to 6MB
  app.use(express.json({ limit: '6mb' }));

  // GET /api/state — Returns current backend state & incident log
  app.get('/api/state', (_req, res) => {
    res.json(serverState);
  });

  // POST /api/analyze-frame — Pre-Trained Multimodal Neural Person & Faint Detector
  app.post('/api/analyze-frame', async (req, res) => {
    try {
      const { imageBase64 } = req.body || {};
      if (!imageBase64 || typeof imageBase64 !== 'string') {
        res.status(400).json({ error: 'Missing imageBase64' });
        return;
      }

      const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: cleanBase64,
              },
            },
            {
              text:
                'Detect every real human person (students, pedestrians, individuals) visible in this video frame. ' +
                'Strictly ignore non-human objects such as basketball hoop poles, backboards, bleachers, doors, walls, or furniture. ' +
                'For each real person, return their tight 2D bounding box [ymin, xmin, ymax, xmax] normalized from 0 to 1000, ' +
                'and set status to "fainted" ONLY if that person is clearly lying collapsed/sprawled horizontally on the floor or ground; ' +
                'otherwise set status to "standing".',
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          temperature: 0.1,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              people: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    box_2d: {
                      type: Type.ARRAY,
                      items: { type: Type.INTEGER },
                      description: '[ymin, xmin, ymax, xmax] normalized 0 to 1000',
                    },
                    status: {
                      type: Type.STRING,
                      description: '"standing" or "fainted"',
                    },
                  },
                  required: ['box_2d', 'status'],
                },
              },
            },
            required: ['people'],
          },
        },
      });

      const rawText = response.text || '{"people":[]}';
      const parsed = JSON.parse(rawText);
      res.json({ people: Array.isArray(parsed.people) ? parsed.people : [] });
    } catch (err: any) {
      res.status(200).json({
        people: [],
        fallback: true,
        error: String(err?.message || err),
      });
    }
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
    serverState.greenWaveActive =
      typeof active === 'boolean' ? active : !serverState.greenWaveActive;
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
      cameraName = 'Place du Méchouar (Test Localization)',
      channel = 'WhatsApp',
      force = false,
    } = req.body || {};

    const now = Date.now();
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

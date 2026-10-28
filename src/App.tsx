import React, { useState, useCallback, useEffect } from 'react';
import {
  HeartPulse,
  Search,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Send,
  Download,
  Code2,
  Smartphone,
} from 'lucide-react';
import {
  OperationalMode,
  TargetClothingColor,
  VideoInputSource,
  DetectedSubject,
  IncidentNotification,
  LAAYOUNE_CAMERA_NODES,
  CameraSectorNode,
  TARGET_COLOR_CONFIG,
} from './types/sahra';
import { VisionCanvas } from './components/VisionCanvas';
import { LaayouneCorridorMap } from './components/LaayouneCorridorMap';
import { PythonCodeInspector } from './components/PythonCodeModal';
import {
  VISION_ENGINE_PY_SOURCE,
  APP_PY_SOURCE,
  REQUIREMENTS_TXT_SOURCE,
} from './data/pythonSource';

export default function App() {
  const [mode, setMode] = useState<OperationalMode>('najda');
  const [targetColor, setTargetColor] = useState<TargetClothingColor>('RED');
  const [inputSource, setInputSource] = useState<VideoInputSource>('simulation');
  const [activeCamera, setActiveCamera] = useState<CameraSectorNode>(LAAYOUNE_CAMERA_NODES[0]);
  const [manualGreenWave, setManualGreenWave] = useState<boolean>(false);
  const [showPythonCode, setShowPythonCode] = useState<boolean>(false);
  const [quotaExceeded, setQuotaExceeded] = useState<boolean>(false);
  const [incidents, setIncidents] = useState<IncidentNotification[]>([
    {
      id: 'inc-101',
      timestamp: '14:22:09',
      mode: 'najda',
      title: 'Medical Emergency — Person Fainted at Place du Méchouar',
      description: 'Hospital → Place du Méchouar route turned GREEN. Opposite cross-traffic routes locked RED.',
      cameraName: 'Place du Méchouar (Test Localization)',
      channel: 'WhatsApp',
      greenWaveTriggered: true,
    },
    {
      id: 'inc-102',
      timestamp: '14:18:41',
      mode: 'tifawt',
      title: 'Missing Person Found — Red Shirt (91%)',
      description: 'Person wearing Red shirt spotted at Boulevard Mekka Crossing.',
      cameraName: 'Boulevard Mekka Crossing',
      channel: 'Telegram',
      greenWaveTriggered: false,
    },
  ]);

  const [telemetry, setTelemetry] = useState<{
    fps: number;
    inferenceMs: number;
    alertActive: boolean;
    alertTitle: string;
    alertDetail: string;
    greenWaveActive: boolean;
    subjects: DetectedSubject[];
  }>({
    fps: 35,
    inferenceMs: 7.4,
    alertActive: false,
    alertTitle: 'All Clear — Normal Activity',
    alertDetail: 'Monitoring Place du Méchouar (Laâyoune). Hospital → Place du Méchouar route registered.',
    greenWaveActive: false,
    subjects: [],
  });

  // Listen for Google Maps Platform Demo Key quota exceeded event
  useEffect(() => {
    const onQuotaExceeded = () => setQuotaExceeded(true);
    window.addEventListener('gmp-quota-exceeded', onQuotaExceeded);
    return () => window.removeEventListener('gmp-quota-exceeded', onQuotaExceeded);
  }, []);

  // Load initial state from Express backend (/api/state)
  useEffect(() => {
    fetch('/api/state')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.incidents) {
          setIncidents(data.incidents);
        }
        if (typeof data?.greenWaveActive === 'boolean') {
          setManualGreenWave(data.greenWaveActive);
        }
      })
      .catch(() => {});
  }, []);

  const handleModeSelect = (newMode: OperationalMode) => {
    setMode(newMode);
    fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode, targetColor }),
    }).catch(() => {});
  };

  const handleColorSelect = (newColor: TargetClothingColor) => {
    setTargetColor(newColor);
    fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, targetColor: newColor }),
    }).catch(() => {});
  };

  const handleTriggerAlert = useCallback(
    (title: string, description: string, forceManual = false) => {
      const channel: 'WhatsApp' | 'Telegram' = mode === 'najda' ? 'WhatsApp' : 'Telegram';
      fetch('/api/dispatch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          title,
          description,
          cameraName: activeCamera.name,
          channel,
          force: forceManual,
        }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((res) => {
          if (res?.state?.incidents) {
            setIncidents(res.state.incidents);
          } else if (!res?.skipped) {
            setIncidents((prev) => [
              {
                id: `inc-${Date.now()}`,
                timestamp: new Date().toTimeString().slice(0, 8),
                mode,
                title,
                description,
                cameraName: activeCamera.name,
                channel,
                greenWaveTriggered: mode === 'najda',
              },
              ...prev.slice(0, 14),
            ]);
          }
        })
        .catch(() => {});
    },
    [mode, activeCamera.name]
  );

  const handleToggleGreenWave = () => {
    const nextState = !manualGreenWave;
    setManualGreenWave(nextState);
    fetch('/api/corridor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: nextState }),
    }).catch(() => {});
  };

  const handleDownloadPythonFiles = () => {
    const files = [
      { name: 'vision_engine.py', content: VISION_ENGINE_PY_SOURCE },
      { name: 'app.py', content: APP_PY_SOURCE },
      { name: 'requirements.txt', content: REQUIREMENTS_TXT_SOURCE },
    ];
    files.forEach((f, idx) => {
      setTimeout(() => {
        const blob = new Blob([f.content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = f.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, idx * 200);
    });
  };

  const effectiveGreenWave = telemetry.greenWaveActive || manualGreenWave;

  return (
    <div className="min-h-screen bg-[#0B0F17] text-[#F8FAFC] flex flex-col">
      {quotaExceeded && (
        <div className="bg-amber-50 border-b border-amber-200 text-amber-900 px-4 py-2.5 text-xs md:text-sm text-center sticky top-0 z-50 shadow-sm">
          <span>
            Google Maps Platform quota reached. If you are the app owner, visit{' '}
            <a
              href="https://developers.google.com/maps/ai/ai-studio?utm_campaign=gmp_mcp_codeassist_v1_aistudio#quota_exceeded_errors"
              target="_blank"
              rel="noopener noreferrer"
              className="underline font-semibold text-amber-950 hover:text-amber-800"
            >
              maps developer site
            </a>{' '}
            for instructions to update your account.
          </span>
        </div>
      )}

      {/* Clean 3-Zone Top Navigation Bar */}
      <header className="flex items-center justify-between gap-8 px-6 py-4 border-b border-[#1E293B] bg-[#0B0F17]">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setShowPythonCode(false);
          }}
          className="text-xl font-bold tracking-tight text-white font-display whitespace-nowrap shrink-0"
        >
          Sahra-Eye Laâyoune
        </a>

        <nav className="hidden md:flex items-center gap-7 text-sm font-medium text-[#94A3B8]">
          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('najda');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 ${
              !showPythonCode && mode === 'najda' ? 'text-white font-semibold' : 'hover:text-white'
            }`}
          >
            1. Medical Emergency
          </button>
          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('tifawt');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 ${
              !showPythonCode && mode === 'tifawt' ? 'text-white font-semibold' : 'hover:text-white'
            }`}
          >
            2. Find Missing Person
          </button>
          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('security');
            }}
            className={`transition-colors whitespace-nowrap shrink-0 ${
              !showPythonCode && mode === 'security' ? 'text-white font-semibold' : 'hover:text-white'
            }`}
          >
            3. Security Guard
          </button>
          <button
            type="button"
            onClick={() => setShowPythonCode((prev) => !prev)}
            className={`transition-colors whitespace-nowrap shrink-0 ${
              showPythonCode ? 'text-[#06B6D4] font-semibold' : 'hover:text-white'
            }`}
          >
            Python Backend Code
          </button>
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleDownloadPythonFiles}
            className="px-4 py-2 text-xs font-semibold text-[#0B0F17] bg-[#06B6D4] rounded-lg hover:bg-[#22D3EE] transition-colors whitespace-nowrap shrink-0"
          >
            Download Python Files
          </button>
        </div>
      </header>

      <main className="flex-1 max-w-[1400px] w-full mx-auto px-6 py-6 space-y-6">
        {/* Step 1: 3 Easy-to-Understand Mode Cards */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('najda');
            }}
            className={`text-left p-4 rounded-xl border transition-all ${
              mode === 'najda' && !showPythonCode
                ? 'bg-[#161F30] border-[#EF4444] shadow-sm'
                : 'bg-[#111827] border-[#1E293B] hover:border-[#334155]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <HeartPulse className="w-4 h-4 text-[#EF4444]" />
                1. Medical Emergency (Najda)
              </span>
              {mode === 'najda' && (
                <span className="text-xs font-semibold text-[#EF4444]">Active</span>
              )}
            </div>
            <p className="text-xs text-[#94A3B8] mt-1.5 leading-relaxed">
              Detects when someone falls or collapses on the street, alerts the ambulance, and turns Boulevard Mekka traffic lights green.
            </p>
          </button>

          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('tifawt');
            }}
            className={`text-left p-4 rounded-xl border transition-all ${
              mode === 'tifawt' && !showPythonCode
                ? 'bg-[#161F30] border-[#F59E0B] shadow-sm'
                : 'bg-[#111827] border-[#1E293B] hover:border-[#334155]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <Search className="w-4 h-4 text-[#F59E0B]" />
                2. Find Missing Person (Tifawt)
              </span>
              {mode === 'tifawt' && (
                <span className="text-xs font-semibold text-[#F59E0B]">
                  Finding: {TARGET_COLOR_CONFIG[targetColor].label}
                </span>
              )}
            </div>
            <p className="text-xs text-[#94A3B8] mt-1.5 leading-relaxed">
              Finds a lost child or senior by scanning city cameras for the color of their shirt (Red, Blue, Yellow, Green, Black, or White).
            </p>
          </button>

          <button
            type="button"
            onClick={() => {
              setShowPythonCode(false);
              handleModeSelect('security');
            }}
            className={`text-left p-4 rounded-xl border transition-all ${
              mode === 'security' && !showPythonCode
                ? 'bg-[#161F30] border-[#06B6D4] shadow-sm'
                : 'bg-[#111827] border-[#1E293B] hover:border-[#334155]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-[#06B6D4]" />
                3. Restricted Area Security
              </span>
              {mode === 'security' && (
                <span className="text-xs font-semibold text-[#06B6D4]">Active</span>
              )}
            </div>
            <p className="text-xs text-[#94A3B8] mt-1.5 leading-relaxed">
              Protects restricted facilities (like the water plant) by detecting intruders wearing ski masks or carrying weapons.
            </p>
          </button>
        </section>

        {/* Optional Python Backend Code Drawer */}
        {showPythonCode ? (
          <PythonCodeInspector onClose={() => setShowPythonCode(false)} />
        ) : (
          /* Main 2-Column Simple Layout */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left 7 Columns: Live Camera / Video Player */}
            <div className="lg:col-span-7 space-y-4">
              <VisionCanvas
                mode={mode}
                targetColor={targetColor}
                inputSource={inputSource}
                activeCamera={activeCamera}
                onInputSourceChange={setInputSource}
                onModeChange={handleModeSelect}
                onTargetColorChange={handleColorSelect}
                onTelemetryUpdate={setTelemetry}
                onTriggerAlert={(title, detail) => handleTriggerAlert(title, detail, false)}
              />

              {/* Simple Engine Speed & Backend Info Bar */}
              <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
                <div className="flex items-center gap-4 text-[#94A3B8]">
                  <span>
                    Speed: <strong className="text-white font-mono">{telemetry.fps} FPS</strong> ({telemetry.inferenceMs} ms)
                  </span>
                  <span>·</span>
                  <span>
                    People in View: <strong className="text-white font-mono">{telemetry.subjects.length}</strong>
                  </span>
                  <span>·</span>
                  <span>
                    Engine: <strong className="text-[#06B6D4]">OpenVINO + YOLOv8 v2.0</strong>
                  </span>
                </div>

                <button
                  type="button"
                  onClick={() => setShowPythonCode(true)}
                  className="flex items-center gap-1.5 text-[#06B6D4] hover:underline font-medium"
                >
                  <Code2 className="w-3.5 h-3.5" />
                  Inspect Upgraded Python Backend
                </button>
              </div>
            </div>

            {/* Right 5 Columns: Clear Status Banner + Live Route Map + Phone Alerts */}
            <div className="lg:col-span-5 space-y-5">
              {/* 1. Big Plain-English Status Card */}
              <div
                className={`p-4 rounded-xl border transition-colors ${
                  telemetry.alertActive
                    ? 'bg-[#EF4444]/15 border-[#EF4444]'
                    : 'bg-[#111827] border-[#1E293B]'
                }`}
              >
                <div className="flex items-start gap-3">
                  {telemetry.alertActive ? (
                    <AlertTriangle className="w-6 h-6 text-[#EF4444] shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle2 className="w-6 h-6 text-[#10B981] shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <div className="text-base font-bold text-white">
                      {telemetry.alertTitle}
                    </div>
                    <p className="text-xs text-[#CBD5E1] mt-1 leading-relaxed">
                      {telemetry.alertDetail}
                    </p>
                  </div>
                </div>
              </div>

              {/* 2. Boulevard Mekka Ambulance Green-Wave Map */}
              <LaayouneCorridorMap
                greenWaveActive={effectiveGreenWave}
                activeCameraId={activeCamera.id}
                onSelectCamera={setActiveCamera}
                onToggleManualGreenWave={handleToggleGreenWave}
              />

              {/* 3. Simple Phone Alerts Feed (WhatsApp & Telegram) */}
              <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden">
                <div className="p-4 border-b border-[#1E293B] flex items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <Smartphone className="w-4 h-4 text-[#10B981]" />
                      Mobile Emergency Alerts (WhatsApp & Telegram)
                    </h3>
                    <p className="text-xs text-[#94A3B8] mt-0.5">
                      Instant phone messages sent to Laâyoune responders
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      handleTriggerAlert(
                        telemetry.alertActive
                          ? telemetry.alertTitle
                          : `Test Alert (${mode.toUpperCase()})`,
                        telemetry.alertActive
                          ? telemetry.alertDetail
                          : `Manual test alert sent from ${activeCamera.name}.`,
                        true
                      )
                    }
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-[#10B981] hover:bg-[#34D399] text-[#0B0F17] rounded-lg transition-colors whitespace-nowrap shrink-0"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Send Test Alert
                  </button>
                </div>

                <div className="divide-y divide-[#1E293B] max-h-[240px] overflow-y-auto">
                  {incidents.map((inc) => (
                    <div key={inc.id} className="p-3.5 text-xs space-y-1 hover:bg-[#161F30]/40">
                      <div className="flex items-center justify-between text-[#94A3B8]">
                        <span className="font-semibold text-[#06B6D4]">
                          {inc.channel} Alert · {inc.cameraName}
                        </span>
                        <span className="font-mono">{inc.timestamp}</span>
                      </div>
                      <div className="font-semibold text-white">{inc.title}</div>
                      <p className="text-[#94A3B8] leading-relaxed">{inc.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

import React, { useState } from 'react';
import { Download, Copy, Check, FileCode, X, Cpu, Zap, Layers } from 'lucide-react';
import {
  VISION_ENGINE_PY_SOURCE,
  APP_PY_SOURCE,
  REQUIREMENTS_TXT_SOURCE,
} from '../data/pythonSource';

interface PythonCodeModalProps {
  onClose?: () => void;
}

export const PythonCodeInspector: React.FC<PythonCodeModalProps> = ({ onClose }) => {
  const [selectedFile, setSelectedFile] = useState<'vision_engine.py' | 'app.py' | 'requirements.txt'>('vision_engine.py');
  const [copied, setCopied] = useState<boolean>(false);

  const getFileContent = (name: 'vision_engine.py' | 'app.py' | 'requirements.txt') => {
    switch (name) {
      case 'vision_engine.py':
        return VISION_ENGINE_PY_SOURCE;
      case 'app.py':
        return APP_PY_SOURCE;
      case 'requirements.txt':
        return REQUIREMENTS_TXT_SOURCE;
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getFileContent(selectedFile));
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadFile = (filename: 'vision_engine.py' | 'app.py' | 'requirements.txt') => {
    const content = getFileContent(filename);
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadAll = () => {
    handleDownloadFile('vision_engine.py');
    setTimeout(() => handleDownloadFile('app.py'), 200);
    setTimeout(() => handleDownloadFile('requirements.txt'), 400);
  };

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-4 bg-[#0B0F17]/60">
        <div>
          <h2 className="text-base font-semibold text-white">
            Upgraded v2.0 Python Backend (`vision_engine.py` & `app.py`)
          </h2>
          <p className="text-xs text-[#94A3B8] mt-0.5">
            Includes Temporal Multi-Frame Fall Verification, CLAHE Desert-Light Normalization, ByteTrack ID Persistence & Threaded Camera Streaming
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleDownloadAll}
            className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold bg-[#06B6D4] hover:bg-[#22D3EE] text-[#0B0F17] rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Download All Python Files
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="p-2 text-[#94A3B8] hover:text-white bg-[#161F30] rounded-lg"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Backend Architecture Improvements Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-4 bg-[#0B0F17]/40 border-b border-[#1E293B] text-xs">
        <div className="p-3 rounded-lg bg-[#161F30]/60 border border-[#1E293B]">
          <div className="font-semibold text-white flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#06B6D4]" />
            1. Anti-Flicker Temporal Buffer
          </div>
          <p className="text-[#94A3B8] mt-1">
            Tracks each person across 12 frames with ByteTrack so bending down or brief camera noise never triggers a false ambulance alarm.
          </p>
        </div>
        <div className="p-3 rounded-lg bg-[#161F30]/60 border border-[#1E293B]">
          <div className="font-semibold text-white flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-[#F59E0B]" />
            2. CLAHE Lighting Normalization
          </div>
          <p className="text-[#94A3B8] mt-1">
            Normalizes harsh desert sunlight and night streetlamp glare in LAB color space before matching shirt colors.
          </p>
        </div>
        <div className="p-3 rounded-lg bg-[#161F30]/60 border border-[#1E293B]">
          <div className="font-semibold text-white flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[#10B981]" />
            3. Threaded Camera & Async Webhooks
          </div>
          <p className="text-[#94A3B8] mt-1">
            Dedicated daemon thread grabs frames without I/O blocking while WhatsApp/Telegram alerts run asynchronously with a 15s cooldown.
          </p>
        </div>
      </div>

      {/* File Tabs */}
      <div className="px-4 py-2.5 bg-[#0B0F17] border-b border-[#1E293B] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          {(['vision_engine.py', 'app.py', 'requirements.txt'] as const).map((fname) => (
            <button
              key={fname}
              type="button"
              onClick={() => setSelectedFile(fname)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition-colors whitespace-nowrap shrink-0 ${
                selectedFile === fname
                  ? 'bg-[#1E293B] text-[#06B6D4] font-semibold'
                  : 'text-[#94A3B8] hover:text-white'
              }`}
            >
              <FileCode className="w-3.5 h-3.5" />
              {fname}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono bg-[#161F30] hover:bg-[#1E293B] text-white rounded-lg transition-colors whitespace-nowrap shrink-0"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-[#10B981]" />
              Copied
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              Copy {selectedFile}
            </>
          )}
        </button>
      </div>

      {/* Code Viewer */}
      <div className="bg-[#0B0F17] p-4 overflow-x-auto max-h-[420px] overflow-y-auto">
        <pre className="text-xs font-mono text-[#E2E8F0] leading-relaxed">
          {getFileContent(selectedFile)}
        </pre>
      </div>
    </div>
  );
};

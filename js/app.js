/**
 * Marine Diesel RPM Meter – App Orchestrator
 * Connects audio engine, DSP processing, and UI components
 */

import { AudioEngine } from './audio-engine.js';
import {
    RPM_RANGES,
    harmonicProductSpectrum,
    findPeakFrequency,
    frequencyToRPM,
    MovingAverage,
    MovingMedian,
    RPMHistory
} from './dsp.js';
import { RPMGauge } from './gauge.js';
import { SpectrumAnalyzer } from './spectrum.js';

// ─── State ───────────────────────────────────────────────────────────────────
let audioEngine = null;
let gauge = null;
let spectrum = null;
let movingAvg = new MovingAverage(5);
let movingMedian = new MovingMedian(7);
let rpmHistory = new RPMHistory();
let animationId = null;
let isRunning = false;
let cylinders = 4;
let currentRange = 'idle'; // 'idle' | 'cruise' | 'high' | 'auto'
let source = 'file'; // 'file' | 'mic'
let spectrumVisible = false;
let historyVisible = false;
let playbackStartTime = 0;
let fileDuration = 0;

// ─── DOM References ──────────────────────────────────────────────────────────
const rpmValueEl = document.getElementById('rpmValue');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const detectedFreqEl = document.getElementById('detectedFreq');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnSourceFile = document.getElementById('btnSourceFile');
const btnSourceMic = document.getElementById('btnSourceMic');
const audioFileInput = document.getElementById('audioFile');
const fileLabel = document.getElementById('fileLabel');
const fileInputGroup = document.getElementById('fileInputGroup');
const progressBarContainer = document.getElementById('progressBarContainer');
const progressBar = document.getElementById('progressBar');
const currentTimeEl = document.getElementById('currentTime');
const totalTimeEl = document.getElementById('totalTime');
const btnToggleSpectrum = document.getElementById('btnToggleSpectrum');
const btnToggleHistory = document.getElementById('btnToggleHistory');
const spectrumSection = document.getElementById('spectrumSection');
const historySection = document.getElementById('historySection');
const cylinderGroup = document.getElementById('cylinderGroup');
const btnExportCSV = document.getElementById('btnExportCSV');
const historyCanvas = document.getElementById('historyCanvas');

// ─── Initialize ──────────────────────────────────────────────────────────────
function init() {
    gauge = new RPMGauge('gaugeCanvas');
    spectrum = new SpectrumAnalyzer('spectrumCanvas');
    gauge.start();

    setupEventListeners();
    registerServiceWorker();
    setupHistoryCanvas();
}

function setupEventListeners() {
    // Source toggle
    btnSourceFile.addEventListener('click', () => setSource('file'));
    btnSourceMic.addEventListener('click', () => setSource('mic'));

    // File input
    audioFileInput.addEventListener('change', handleFileSelect);

    // Cylinder selection
    cylinderGroup.addEventListener('click', (e) => {
        const btn = e.target.closest('.cyl-btn');
        if (!btn) return;
        cylinders = parseInt(btn.dataset.cyl, 10);
        cylinderGroup.querySelectorAll('.cyl-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });

    // Range selection
    const rangeGroup = document.getElementById('rangeGroup');
    if (rangeGroup) {
        rangeGroup.addEventListener('click', (e) => {
            const btn = e.target.closest('.range-btn');
            if (!btn) return;
            currentRange = btn.dataset.range;
            rangeGroup.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (audioEngine && RPM_RANGES[currentRange]) {
                audioEngine.setFilterCutoff(RPM_RANGES[currentRange].filterCutoff);
            }
        });
    }

    // Start/Stop
    btnStart.addEventListener('click', handleStart);
    btnStop.addEventListener('click', handleStop);

    // Spectrum toggle
    btnToggleSpectrum.addEventListener('click', () => {
        spectrumVisible = !spectrumVisible;
        spectrumSection.style.display = spectrumVisible ? 'block' : 'none';
        btnToggleSpectrum.classList.toggle('active', spectrumVisible);
        if (spectrumVisible) spectrum.resize();
    });

    // History toggle
    btnToggleHistory.addEventListener('click', () => {
        historyVisible = !historyVisible;
        historySection.style.display = historyVisible ? 'block' : 'none';
        btnToggleHistory.classList.toggle('active', historyVisible);
        if (historyVisible) drawHistory();
    });

    // CSV export
    btnExportCSV.addEventListener('click', exportCSV);
}

// ─── Source Selection ────────────────────────────────────────────────────────
function setSource(newSource) {
    source = newSource;
    btnSourceFile.classList.toggle('active', source === 'file');
    btnSourceMic.classList.toggle('active', source === 'mic');
    fileInputGroup.style.display = source === 'file' ? 'flex' : 'none';
}

// ─── File Handling ───────────────────────────────────────────────────────────
async function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    fileLabel.textContent = file.name;
    fileLabel.classList.add('has-file');

    try {
        if (!audioEngine) {
            audioEngine = new AudioEngine();
            await audioEngine.init();
        }
        const result = await audioEngine.loadFile(file);
        fileDuration = result.duration;
        totalTimeEl.textContent = formatTime(fileDuration);
        setStatus('Datei geladen', false);
    } catch (err) {
        setStatus('Fehler: ' + err.message, false, true);
        console.error('File load error:', err);
    }
}

// ─── Start/Stop ──────────────────────────────────────────────────────────────
async function handleStart() {
    try {
        if (!audioEngine) {
            audioEngine = new AudioEngine();
            await audioEngine.init();
        }

        // Apply range-specific filter cutoff
        const rangeConfig = RPM_RANGES[currentRange] || RPM_RANGES.idle;
        audioEngine.setFilterCutoff(rangeConfig.filterCutoff);

        // Reset history and filters for new recording
        rpmHistory.clear();
        movingAvg.reset();
        movingMedian.reset();

        if (source === 'file') {
            if (!audioEngine.duration) {
                setStatus('Bitte zuerst eine Datei wählen', false, true);
                return;
            }
            playbackStartTime = audioEngine.audioContext.currentTime;
            progressBarContainer.style.display = 'block';

            const playPromise = audioEngine.playFile();
            setRunning(true);
            startAnalysisLoop();

            // Auto-stop when playback finishes
            playPromise.then(() => {
                handleStop();
            });
        } else {
            await audioEngine.startMicrophone();
            playbackStartTime = audioEngine.audioContext.currentTime;
            setRunning(true);
            startAnalysisLoop();
        }

        // Try to acquire wake lock to keep screen on
        acquireWakeLock();
    } catch (err) {
        setStatus('Fehler: ' + err.message, false, true);
        console.error('Start error:', err);
    }
}

function handleStop() {
    if (source === 'file') {
        try { audioEngine.stopFile(); } catch (e) { /* already stopped */ }
    } else {
        audioEngine.stopMicrophone();
    }
    stopAnalysisLoop();
    setRunning(false);
    releaseWakeLock();
}

function setRunning(running) {
    isRunning = running;
    btnStart.style.display = running ? 'none' : 'flex';
    btnStop.style.display = running ? 'flex' : 'none';
    setStatus(running ? 'Messung läuft…' : 'Gestoppt', running);
}

// ─── Analysis Loop ───────────────────────────────────────────────────────────
let debugFrameCount = 0;

function startAnalysisLoop() {
    debugFrameCount = 0;

    function loop() {
        if (!isRunning) return;

        // Get frequency data from the audio engine
        const freqData = audioEngine.getFrequencyData();
        const sampleRate = audioEngine.sampleRate;
        const fftSize = audioEngine.fftSize;

        // Get range configuration
        const rangeConfig = RPM_RANGES[currentRange] || RPM_RANGES.idle;
        const minFreq = (rangeConfig.minRPM * cylinders) / 120;
        const maxFreq = (rangeConfig.maxRPM * cylinders) / 120;

        // Apply HPS algorithm to find the fundamental frequency
        const hpsResult = harmonicProductSpectrum(freqData, rangeConfig.numHarmonics, rangeConfig.power);
        const peak = findPeakFrequency(hpsResult, sampleRate, fftSize, minFreq, maxFreq);

        // Convert frequency to RPM
        let rpm = 0;
        if (peak.frequency > 0) {
            rpm = frequencyToRPM(peak.frequency, cylinders);
        }

        // Apply rolling median filter for rock-solid stability
        const smoothedRPM = rpm > 0 ? movingMedian.push(rpm) : movingMedian.getMedian();

        if (peak.frequency > 0) {
            // --- DEBUG OUTPUT ---
            let dbgEl = document.getElementById('ag-debug-info');
            if (!dbgEl) {
                dbgEl = document.createElement('div');
                dbgEl.id = 'ag-debug-info';
                dbgEl.style = 'position:fixed; bottom:10px; left:10px; background:black; color:lime; font-family:monospace; padding:5px; z-index:9999; font-size:12px;';
                document.body.appendChild(dbgEl);
            }
            dbgEl.innerText = `DEBUG: Bin=${peak.binIndex} Freq=${peak.frequency.toFixed(2)}Hz RawRPM=${rpm} SmthRPM=${Math.round(smoothedRPM)} Val=${peak.magnitude.toFixed(1)}dB`;
            // --------------------
            debugFrameCount++;
        }

        // Always update UI display
        updateRPMDisplay(Math.round(smoothedRPM), peak.frequency);
        gauge.setRPM(smoothedRPM);

        // Record to history timeline (only once!)
        const elapsed = audioEngine.audioContext.currentTime - playbackStartTime;
        rpmHistory.record(Math.round(smoothedRPM), elapsed);

        // Update spectrum if visible
        if (spectrumVisible) {
            spectrum.draw(freqData, sampleRate, fftSize, peak.frequency > 0 ? peak.frequency : null);
        }

        // Update history graph if visible
        if (historyVisible) {
            drawHistory();
        }

        // Update progress bar for file mode
        if (source === 'file' && fileDuration > 0) {
            const progress = Math.min(elapsed / fileDuration * 100, 100);
            progressBar.style.width = progress + '%';
            currentTimeEl.textContent = formatTime(elapsed);
        }

        animationId = requestAnimationFrame(loop);
    }

    animationId = requestAnimationFrame(loop);
}

function stopAnalysisLoop() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

// ─── UI Updates ──────────────────────────────────────────────────────────────
function updateRPMDisplay(rpm, frequency) {
    const displayRPM = Math.round(rpm);
    rpmValueEl.textContent = displayRPM;
    gauge.setRPM(displayRPM);
    detectedFreqEl.textContent = frequency > 0
        ? frequency.toFixed(1) + ' Hz'
        : '— Hz';
}

function setStatus(text, active, error = false) {
    statusText.textContent = text;
    statusDot.className = 'status-dot' + (active ? ' active' : '') + (error ? ' error' : '');
}

// ─── History Graph ───────────────────────────────────────────────────────────
let historyCtx = null;

function setupHistoryCanvas() {
    const canvas = historyCanvas;
    if (!canvas) return;
    historyCtx = canvas.getContext('2d');

    const observer = new ResizeObserver(() => {
        const rect = canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        historyCtx.scale(dpr, dpr);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        if (historyVisible) drawHistory();
    });
    observer.observe(canvas.parentElement);
}

function drawHistory() {
    if (!historyCtx || !historyCanvas) return;

    const data = rpmHistory.getData();
    const dpr = window.devicePixelRatio || 1;
    const w = historyCanvas.width / dpr;
    const h = historyCanvas.height / dpr;

    const padding = { top: 20, right: 15, bottom: 30, left: 45 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    historyCtx.clearRect(0, 0, w, h);

    // Background
    historyCtx.fillStyle = '#f5f5f7';
    historyCtx.fillRect(0, 0, w, h);

    // Grid and labels
    historyCtx.strokeStyle = '#e0e0e0';
    historyCtx.fillStyle = '#6b7280';
    historyCtx.font = '10px -apple-system, sans-serif';
    historyCtx.textAlign = 'right';
    historyCtx.lineWidth = 0.5;

    let actualMax = 100;
    for (const d of data) {
        if (d.rpm > actualMax) actualMax = d.rpm;
    }
    
    // Scale up to nearest 500
    const maxRPM = Math.ceil(actualMax / 500) * 500;
    const rpmSteps = [0, maxRPM * 0.25, maxRPM * 0.5, maxRPM * 0.75, maxRPM];

    for (const rpm of rpmSteps) {
        const y = padding.top + plotH - (rpm / maxRPM) * plotH;
        historyCtx.beginPath();
        historyCtx.moveTo(padding.left, y);
        historyCtx.lineTo(w - padding.right, y);
        historyCtx.stroke();
        historyCtx.fillText(Math.round(rpm).toString(), padding.left - 5, y + 3);
    }

    if (data.length < 2) {
        historyCtx.fillStyle = '#6b7280';
        historyCtx.textAlign = 'center';
        historyCtx.font = '12px -apple-system, sans-serif';
        historyCtx.fillText('Noch keine Daten', w / 2, h / 2);
        return;
    }

    // --- Statistics Update ---
    let sumRpm = 0;
    let secBuckets = {}; // Group by floor(time)
    for (const d of data) {
        sumRpm += d.rpm;
        const sec = Math.floor(d.time);
        if (!secBuckets[sec]) secBuckets[sec] = [];
        secBuckets[sec].push(d.rpm);
    }
    const avgRpm = Math.round(sumRpm / data.length);
    document.getElementById('avgRpmDisplay').innerText = avgRpm;

    let secHtml = '';
    for (const sec in secBuckets) {
        const vals = secBuckets[sec];
        const secAvg = Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
        secHtml += `<div>Sekunde ${sec}: <strong>${secAvg} RPM</strong></div>`;
    }
    document.getElementById('secRpmList').innerHTML = secHtml || 'Keine Daten';
    // -------------------------

    // Time axis
    const maxTime = data[data.length - 1].time;
    historyCtx.textAlign = 'center';
    const numTimeLabels = Math.min(6, Math.floor(maxTime));
    for (let i = 0; i <= numTimeLabels; i++) {
        const t = (maxTime / numTimeLabels) * i;
        const x = padding.left + (t / maxTime) * plotW;
        historyCtx.fillStyle = '#6b7280';
        historyCtx.fillText(formatTime(t), x, h - 8);
    }

    // Draw RPM line
    historyCtx.beginPath();
    historyCtx.strokeStyle = '#0066cc';
    historyCtx.lineWidth = 2;
    historyCtx.lineJoin = 'round';

    for (let i = 0; i < data.length; i++) {
        const x = padding.left + (data[i].time / maxTime) * plotW;
        const y = padding.top + plotH - (data[i].rpm / maxRPM) * plotH;
        if (i === 0) {
            historyCtx.moveTo(x, y);
        } else {
            historyCtx.lineTo(x, y);
        }
    }
    historyCtx.stroke();

    // Fill area under the curve
    historyCtx.lineTo(padding.left + plotW, padding.top + plotH);
    historyCtx.lineTo(padding.left, padding.top + plotH);
    historyCtx.closePath();
    historyCtx.fillStyle = 'rgba(0, 102, 204, 0.08)';
    historyCtx.fill();
}

// ─── CSV Export ──────────────────────────────────────────────────────────────
function exportCSV() {
    const csv = rpmHistory.getCSV();
    if (!csv || rpmHistory.getData().length === 0) {
        setStatus('Keine Daten zum Exportieren', false, true);
        return;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rpm_messung_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setStatus('CSV exportiert', false);
}

// ─── Wake Lock ───────────────────────────────────────────────────────────────
let wakeLock = null;

async function acquireWakeLock() {
    if ('wakeLock' in navigator) {
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.warn('Wake Lock not available:', err);
        }
    }
}

function releaseWakeLock() {
    if (wakeLock) {
        wakeLock.release();
        wakeLock = null;
    }
}

// ─── Utilities ───────────────────────────────────────────────────────────────
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + s.toString().padStart(2, '0');
}

// ─── Service Worker ──────────────────────────────────────────────────────────
function registerServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(function(registrations) {
            for(let registration of registrations) {
                registration.unregister();
                console.log('Service Worker unregistered to clear cache');
            }
        });
    }
}

// ─── Boot ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);

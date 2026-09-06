/* ==========================================================================
   Video RPM Analyzer – Client-Side JavaScript  (v1.2.0)
   Ports the Python FFT-based flywheel stripe detection to the browser.
   No server needed — runs entirely on the device.
   ========================================================================== */

'use strict';

const APP_VERSION = '1.2.0';

// ---------------------------------------------------------------------------
// FFT – Cooley-Tukey radix-2 (in-place)
// ---------------------------------------------------------------------------

function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

/**
 * In-place radix-2 FFT.
 * @param {Float64Array} re  Real parts (length must be power of 2).
 * @param {Float64Array} im  Imaginary parts (same length).
 */
function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const halfLen = len >> 1;
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang);
        const wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let curRe = 1, curIm = 0;
            for (let j = 0; j < halfLen; j++) {
                const a = i + j;
                const b = a + halfLen;
                const tRe = re[b] * curRe - im[b] * curIm;
                const tIm = re[b] * curIm + im[b] * curRe;
                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;
                const newCurRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newCurRe;
            }
        }
    }
}

/**
 * Compute the magnitude spectrum of a real signal.
 */
function rfftMagnitude(signal, sampleRate) {
    const N = nextPow2(signal.length);
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let i = 0; i < signal.length; i++) {
        const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (signal.length - 1)));
        re[i] = signal[i] * w;
    }
    fft(re, im);
    const halfN = N / 2 + 1;
    const freqs = new Float64Array(halfN);
    const magnitudes = new Float64Array(halfN);
    for (let i = 0; i < halfN; i++) {
        freqs[i] = i * sampleRate / N;
        magnitudes[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return { freqs, magnitudes };
}

// ---------------------------------------------------------------------------
// RPM Analysis Engine
// ---------------------------------------------------------------------------

function computeROIPositions(cx, cy, radius, nROIs = 8, ringFraction = 0.78) {
    const r = radius * ringFraction;
    const positions = [];
    for (let i = 0; i < nROIs; i++) {
        const angle = (2 * Math.PI * i) / nROIs;
        positions.push({
            x: Math.round(cx + r * Math.cos(angle)),
            y: Math.round(cy + r * Math.sin(angle)),
        });
    }
    return positions;
}

function extractBrightness(imageData, positions, patchHalf = 10) {
    const { data, width, height } = imageData;
    const values = [];
    for (const { x, y } of positions) {
        let sum = 0, count = 0;
        const x0 = Math.max(0, x - patchHalf);
        const x1 = Math.min(width - 1, x + patchHalf);
        const y0 = Math.max(0, y - patchHalf);
        const y1 = Math.min(height - 1, y + patchHalf);
        for (let py = y0; py <= y1; py++) {
            for (let px = x0; px <= x1; px++) {
                const idx = (py * width + px) * 4;
                sum += 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                count++;
            }
        }
        values.push(count > 0 ? sum / count : 0);
    }
    return values;
}

function windowedFFT(signal, fps, windowSec = 2.0, stepSec = 0.25,
                     minRPM = 200, maxRPM = 3000) {
    const windowFrames = Math.round(windowSec * fps);
    const stepFrames = Math.max(1, Math.round(stepSec * fps));
    const nyquistRPM = (fps / 2) * 60;
    maxRPM = Math.min(maxRPM, nyquistRPM * 0.95);
    const minFreq = minRPM / 60, maxFreq = maxRPM / 60;
    const times = [], rpms = [], confidences = [];

    for (let i = 0; i + windowFrames <= signal.length; i += stepFrames) {
        const chunk = signal.slice(i, i + windowFrames);
        let mean = 0;
        for (let j = 0; j < chunk.length; j++) mean += chunk[j];
        mean /= chunk.length;
        for (let j = 0; j < chunk.length; j++) chunk[j] -= mean;

        const { freqs, magnitudes } = rfftMagnitude(chunk, fps);
        let peakMag = 0, peakFreq = 0, sumMag = 0, countInRange = 0;
        for (let j = 0; j < freqs.length; j++) {
            if (freqs[j] >= minFreq && freqs[j] <= maxFreq) {
                sumMag += magnitudes[j];
                countInRange++;
                if (magnitudes[j] > peakMag) {
                    peakMag = magnitudes[j];
                    peakFreq = freqs[j];
                }
            }
        }
        const meanMag = countInRange > 0 ? sumMag / countInRange : 1;
        times.push((i + windowFrames / 2) / fps);
        rpms.push(peakFreq * 60);
        confidences.push(peakMag / (meanMag + 1e-10));
    }
    return { times, rpms, confidences };
}

function combineResults(allTraces) {
    const nWindows = allTraces[0].rpms.length;
    const bestRPMs = new Float64Array(nWindows);
    for (let t = 0; t < nWindows; t++) {
        let bestConf = -1, bestRPM = 0;
        for (const trace of allTraces) {
            if (trace.confidences[t] > bestConf) {
                bestConf = trace.confidences[t];
                bestRPM = trace.rpms[t];
            }
        }
        bestRPMs[t] = bestRPM;
    }
    const sorted = Array.from(bestRPMs).sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    let filtered = iqr > 0
        ? sorted.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr)
        : sorted;
    const overallRPM = filtered.length > 0
        ? filtered[Math.floor(filtered.length / 2)]
        : sorted[Math.floor(sorted.length / 2)];
    return { bestRPMs: Array.from(bestRPMs), overallRPM };
}

// ---------------------------------------------------------------------------
// Peak Detection – find individual stripe passages
// ---------------------------------------------------------------------------

/**
 * Detect brightness peaks (stripe passing the ROI point).
 * Returns array of { index, time, brightness }.
 */
function detectPeaks(signal, fps, maxRPM = 3000) {
    const n = signal.length;
    if (n < 5) return [];

    // Minimum distance between peaks based on max plausible RPM
    const minDist = Math.max(2, Math.floor((60 / maxRPM) * fps * 0.6));

    // Smooth with 3-point moving average
    const smooth = new Float64Array(n);
    smooth[0] = signal[0];
    smooth[n - 1] = signal[n - 1];
    for (let i = 1; i < n - 1; i++) {
        smooth[i] = (signal[i - 1] + signal[i] + signal[i + 1]) / 3;
    }

    // Adaptive threshold: mean + fraction of (max - mean)
    let sum = 0, maxVal = -Infinity;
    for (let i = 0; i < n; i++) {
        sum += smooth[i];
        if (smooth[i] > maxVal) maxVal = smooth[i];
    }
    const mean = sum / n;
    const threshold = mean + 0.25 * (maxVal - mean);

    // Find local maxima above threshold
    const candidates = [];
    for (let i = 1; i < n - 1; i++) {
        if (smooth[i] > threshold &&
            smooth[i] >= smooth[i - 1] &&
            smooth[i] >= smooth[i + 1]) {
            candidates.push({ index: i, value: smooth[i] });
        }
    }

    // Enforce minimum distance (keep highest in each window)
    const peaks = [];
    for (const c of candidates) {
        if (peaks.length === 0 || c.index - peaks[peaks.length - 1].index >= minDist) {
            peaks.push(c);
        } else if (c.value > peaks[peaks.length - 1].value) {
            peaks[peaks.length - 1] = c;
        }
    }

    return peaks.map(p => ({
        index: p.index,
        time: p.index / fps,
        brightness: signal[p.index],
    }));
}

/**
 * Compute per-rotation RPM from peak intervals.
 * Returns { rotations, peakCountRPM }.
 */
function analyzeRotations(peaks, fps) {
    const rotations = [];
    for (let i = 0; i < peaks.length; i++) {
        const rot = {
            num: i + 1,
            frame: peaks[i].index,
            time: peaks[i].time,
            period: null,
            rpm: null,
        };
        if (i > 0) {
            rot.period = peaks[i].time - peaks[i - 1].time;
            rot.rpm = rot.period > 0 ? 60 / rot.period : 0;
        }
        rotations.push(rot);
    }

    // Overall peak-counting RPM
    let peakCountRPM = 0;
    if (peaks.length >= 2) {
        const span = peaks[peaks.length - 1].time - peaks[0].time;
        if (span > 0) {
            peakCountRPM = ((peaks.length - 1) / span) * 60;
        }
    }

    return { rotations, peakCountRPM };
}

// ---------------------------------------------------------------------------
// Filmstrip – capture thumbnail frames for visual proof
// ---------------------------------------------------------------------------

/**
 * Capture a filmstrip of consecutive frame thumbnails covering ~2 rotations.
 * Starts from a peak near the middle of the video.
 * Returns array of { canvas, frameIndex, time, isPeak }.
 */
async function captureFilmstrip(video, peaks, fps, overallRPM,
                                cx, cy, radius, roiPositions,
                                onProgress, thumbWidth = 120) {
    const framesPerRot = (60 / overallRPM) * fps;
    // Show 2-4 rotations, capped between 8 and 24 thumbnails
    const targetRots = framesPerRot < 5 ? 4 : (framesPerRot < 10 ? 3 : 2);
    const numFrames = Math.min(24, Math.max(8, Math.ceil(framesPerRot * targetRots)));

    // Find a peak near the middle of the video
    const midTime = video.duration / 2;
    let startPeak = peaks.length > 0 ? peaks[0] : { index: Math.floor(fps * midTime), time: midTime };
    let bestDist = Infinity;
    for (const p of peaks) {
        const dist = Math.abs(p.time - midTime);
        if (dist < bestDist) { bestDist = dist; startPeak = p; }
    }

    // Start 2 frames before the peak so we see the approach
    const startFrame = Math.max(0, startPeak.index - 2);
    const peakSet = new Set(peaks.map(p => p.index));

    const thumbH = Math.round(thumbWidth * (video.videoHeight / video.videoWidth));
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = video.videoWidth;
    tmpCanvas.height = video.videoHeight;
    const tmpCtx = tmpCanvas.getContext('2d');
    const thumbnails = [];

    for (let i = 0; i < numFrames; i++) {
        const frameIdx = startFrame + i;
        const t = frameIdx / fps;
        if (t > video.duration) break;

        await seekVideo(video, t);
        tmpCtx.drawImage(video, 0, 0);

        // Draw flywheel circle overlay
        tmpCtx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
        tmpCtx.lineWidth = 4;
        tmpCtx.beginPath();
        tmpCtx.arc(cx, cy, radius, 0, 2 * Math.PI);
        tmpCtx.stroke();

        // Draw ROI points
        tmpCtx.fillStyle = 'rgba(255, 255, 0, 0.8)';
        for (const pos of roiPositions) {
            tmpCtx.fillRect(pos.x - 6, pos.y - 6, 12, 12);
        }

        // Check if near a peak (stripe visible)
        let isPeak = false;
        for (const p of peaks) {
            if (Math.abs(p.index - frameIdx) <= 1) { isPeak = true; break; }
        }

        // Mark peak frames with a red banner
        if (isPeak) {
            const bannerH = Math.round(video.videoHeight * 0.04);
            tmpCtx.fillStyle = 'rgba(255, 50, 50, 0.85)';
            tmpCtx.fillRect(0, 0, video.videoWidth, bannerH);
            tmpCtx.fillStyle = 'white';
            tmpCtx.font = `bold ${Math.round(bannerH * 0.7)}px sans-serif`;
            tmpCtx.textAlign = 'center';
            tmpCtx.fillText('★ STRIPE DETECTED', video.videoWidth / 2, bannerH * 0.75);
            tmpCtx.textAlign = 'left';
        }

        // Downscale to thumbnail
        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = thumbWidth;
        thumbCanvas.height = thumbH;
        thumbCanvas.getContext('2d').drawImage(tmpCanvas, 0, 0, thumbWidth, thumbH);

        thumbnails.push({ canvas: thumbCanvas, frameIndex: frameIdx, time: t, isPeak });
        if (onProgress) onProgress(i + 1, numFrames);
    }

    return thumbnails;
}

// ---------------------------------------------------------------------------
// Video Frame Extraction & Seeking
// ---------------------------------------------------------------------------

function seekVideo(video, timeSec) {
    return new Promise((resolve) => {
        if (Math.abs(video.currentTime - timeSec) < 0.001) { resolve(); return; }
        const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve(); };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = timeSec;
        setTimeout(() => { video.removeEventListener('seeked', onSeeked); resolve(); }, 500);
    });
}

async function extractFrames(video, canvas, roiPositions, fps, onProgress) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const duration = video.duration;
    const dt = 1.0 / fps;
    const totalFrames = Math.floor(duration * fps);
    const nROIs = roiPositions.length;
    const signals = Array.from({ length: nROIs }, () => []);
    let extractedCount = 0;

    for (let t = 0; t < duration - dt / 2; t += dt) {
        await seekVideo(video, t);
        ctx.drawImage(video, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const brightness = extractBrightness(imageData, roiPositions);
        for (let j = 0; j < nROIs; j++) signals[j].push(brightness[j]);
        extractedCount++;
        if (onProgress) onProgress(extractedCount, totalFrames);
        if (extractedCount % 20 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    return signals;
}

// ---------------------------------------------------------------------------
// Chart Drawing
// ---------------------------------------------------------------------------

function drawLineChart(canvas, xData, yData, {
    color = '#0066cc', fillColor = 'rgba(0, 102, 204, 0.1)',
    lineWidth = 2, xLabel = '', yMin = null, yMax = null,
    highlightY = null, highlightLabel = '',
    peakAnnotation = null,  // { freq, rpm } for FFT annotation
} = {}) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = { top: 12, right: 12, bottom: 28, left: 48 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const xMin = Math.min(...xData), xMax = Math.max(...xData);
    if (yMin === null) yMin = Math.min(...yData) * 0.9;
    if (yMax === null) yMax = Math.max(...yData) * 1.1;
    if (yMax <= yMin) yMax = yMin + 1;
    const toX = (v) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const toY = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    ctx.fillStyle = '#f8f9fa'; ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const yVal = yMin + (yMax - yMin) * i / 4;
        const py = toY(yVal);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(W - pad.right, py); ctx.stroke();
        ctx.fillStyle = '#9ca3af'; ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right'; ctx.fillText(Math.round(yVal).toString(), pad.left - 6, py + 3);
    }

    // Highlight line
    if (highlightY !== null) {
        const hy = toY(highlightY);
        ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(pad.left, hy); ctx.lineTo(W - pad.right, hy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#4caf50'; ctx.font = 'bold 11px -apple-system, sans-serif';
        ctx.textAlign = 'left'; ctx.fillText(highlightLabel, pad.left + 4, hy - 4);
    }

    // Data line
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < xData.length; i++) {
        const px = toX(xData[i]), py = toY(yData[i]);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Fill
    if (fillColor) {
        ctx.fillStyle = fillColor; ctx.beginPath();
        ctx.moveTo(toX(xData[0]), toY(yData[0]));
        for (let i = 1; i < xData.length; i++) ctx.lineTo(toX(xData[i]), toY(yData[i]));
        ctx.lineTo(toX(xData[xData.length - 1]), toY(yMin));
        ctx.lineTo(toX(xData[0]), toY(yMin));
        ctx.closePath(); ctx.fill();
    }

    // Peak annotation (for FFT chart)
    if (peakAnnotation) {
        const px = toX(peakAnnotation.rpm);
        const peakIdx = xData.reduce((best, v, i) =>
            Math.abs(v - peakAnnotation.rpm) < Math.abs(xData[best] - peakAnnotation.rpm) ? i : best, 0);
        const py = toY(yData[peakIdx]);
        // Arrow + label
        ctx.fillStyle = '#e53935';
        ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();
        ctx.font = 'bold 11px -apple-system, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${peakAnnotation.freq.toFixed(1)} Hz → ${Math.round(peakAnnotation.rpm)} RPM`,
            px + 8, py - 6);
    }

    // X-axis labels
    ctx.fillStyle = '#9ca3af'; ctx.font = '10px -apple-system, sans-serif'; ctx.textAlign = 'center';
    const nLabelsX = Math.min(6, xData.length);
    for (let i = 0; i < nLabelsX; i++) {
        const idx = Math.round(i * (xData.length - 1) / (nLabelsX - 1));
        ctx.fillText(xData[idx].toFixed(1), toX(xData[idx]), H - 4);
    }
    if (xLabel) {
        ctx.fillStyle = '#6b7280'; ctx.font = '11px -apple-system, sans-serif';
        ctx.fillText(xLabel, W / 2, H - 2);
    }
}

/**
 * Draw an annotated brightness signal with detected peaks marked.
 * Shows a ~2-second window from the middle of the video.
 */
function drawAnnotatedBrightness(canvas, signal, fps, peaks, overallRPM) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = { top: 20, right: 12, bottom: 28, left: 48 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Pick a 2-second window from the middle, centered on a peak
    const windowFrames = Math.round(2.0 * fps);
    const midIdx = Math.floor(signal.length / 2);
    // Find closest peak to middle
    let centerIdx = midIdx;
    let bestDist = Infinity;
    for (const p of peaks) {
        const d = Math.abs(p.index - midIdx);
        if (d < bestDist) { bestDist = d; centerIdx = p.index; }
    }
    const startIdx = Math.max(0, centerIdx - Math.floor(windowFrames / 2));
    const endIdx = Math.min(signal.length, startIdx + windowFrames);
    const windowSignal = signal.slice(startIdx, endIdx);
    const windowPeaks = peaks.filter(p => p.index >= startIdx && p.index < endIdx);

    // Ranges
    const xMin = startIdx / fps, xMax = (endIdx - 1) / fps;
    let yMin = Infinity, yMax = -Infinity;
    for (const v of windowSignal) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    const yPad = (yMax - yMin) * 0.15;
    yMin -= yPad; yMax += yPad;
    if (yMax <= yMin) yMax = yMin + 1;

    const toX = (t) => pad.left + ((t - xMin) / (xMax - xMin || 1)) * plotW;
    const toY = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Background
    ctx.fillStyle = '#f8f9fa'; ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const yVal = yMin + (yMax - yMin) * i / 4;
        const py = toY(yVal);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(W - pad.right, py); ctx.stroke();
        ctx.fillStyle = '#9ca3af'; ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right'; ctx.fillText(Math.round(yVal).toString(), pad.left - 6, py + 3);
    }

    // Threshold line
    let sumSig = 0, maxSig = -Infinity;
    for (const v of windowSignal) { sumSig += v; if (v > maxSig) maxSig = v; }
    const meanSig = sumSig / windowSignal.length;
    const thresh = meanSig + 0.25 * (maxSig - meanSig);
    ctx.strokeStyle = 'rgba(156, 163, 175, 0.6)'; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const ty = toY(thresh);
    ctx.beginPath(); ctx.moveTo(pad.left, ty); ctx.lineTo(W - pad.right, ty); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#9ca3af'; ctx.font = '9px -apple-system, sans-serif';
    ctx.textAlign = 'left'; ctx.fillText('threshold', pad.left + 2, ty - 3);

    // Signal line
    ctx.strokeStyle = '#0066cc'; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < windowSignal.length; i++) {
        const t = (startIdx + i) / fps;
        const px = toX(t), py = toY(windowSignal[i]);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Peak markers
    let rotNum = 0;
    // Find the number of the first peak in the window relative to all peaks
    const firstPeakGlobal = peaks.findIndex(p => p.index >= startIdx);
    for (const p of windowPeaks) {
        const globalIdx = peaks.indexOf(p);
        rotNum = globalIdx + 1;
        const px = toX(p.time);
        const py = toY(signal[p.index]);

        // Vertical dashed line
        ctx.strokeStyle = 'rgba(229, 57, 53, 0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
        ctx.setLineDash([]);

        // Red dot
        ctx.fillStyle = '#e53935';
        ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();

        // Rotation number label
        ctx.fillStyle = '#e53935'; ctx.font = 'bold 10px -apple-system, sans-serif';
        ctx.textAlign = 'center'; ctx.fillText(`#${rotNum}`, px, pad.top - 4);
    }

    // X-axis labels
    ctx.fillStyle = '#9ca3af'; ctx.font = '10px -apple-system, sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
        const t = xMin + (xMax - xMin) * i / 4;
        ctx.fillText(t.toFixed(2) + 's', toX(t), H - 4);
    }
}

// ---------------------------------------------------------------------------
// Canvas Flywheel Selection UI
// ---------------------------------------------------------------------------

class FlywheelSelector {
    constructor(canvasEl, wrapperEl) {
        this.canvas = canvasEl;
        this.wrapper = wrapperEl;
        this.ctx = canvasEl.getContext('2d');
        this.frameImage = null;
        this.center = null;
        this.radius = 200;
        this.roiPositions = [];
        this.imageW = 0;
        this.imageH = 0;
        this._onSelect = null;
        this.wrapper.addEventListener('pointerdown', (e) => this._handleTap(e));
    }

    async showFrame(video) {
        const vw = video.videoWidth, vh = video.videoHeight;
        if (vw === 0 || vh === 0) return;
        this.imageW = vw; this.imageH = vh;
        this.canvas.width = vw; this.canvas.height = vh;
        try {
            this.frameImage = await createImageBitmap(video);
        } catch (_) {
            const tmp = document.createElement('canvas');
            tmp.width = vw; tmp.height = vh;
            tmp.getContext('2d').drawImage(video, 0, 0);
            this.frameImage = tmp;
        }
        this.center = null; this.roiPositions = [];
        this.ctx.drawImage(this.frameImage, 0, 0);
    }

    _handleTap(e) {
        e.preventDefault();
        if (!this.frameImage) return;
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        const scaleX = this.imageW / rect.width;
        const scaleY = this.imageH / rect.height;
        const imgX = Math.round((e.clientX - rect.left) * scaleX);
        const imgY = Math.round((e.clientY - rect.top) * scaleY);
        if (imgX < 0 || imgX >= this.imageW || imgY < 0 || imgY >= this.imageH) return;
        this.center = { x: imgX, y: imgY };
        this._updateROIs(); this.redraw();
        if (this._onSelect) this._onSelect(this.center, this.radius);
    }

    setRadius(r) {
        this.radius = r;
        if (this.center) { this._updateROIs(); this.redraw(); }
    }

    _updateROIs() {
        if (!this.center) return;
        this.roiPositions = computeROIPositions(this.center.x, this.center.y, this.radius, 8, 0.78);
    }

    redraw() {
        const { ctx, frameImage, center, radius } = this;
        if (!frameImage) return;
        ctx.drawImage(frameImage, 0, 0);
        if (!center) return;

        ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = 'red';
        ctx.beginPath(); ctx.arc(center.x, center.y, 5, 0, 2 * Math.PI); ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.9)'; ctx.lineWidth = 2;
        for (let i = 0; i < this.roiPositions.length; i++) {
            const { x, y } = this.roiPositions[i];
            ctx.strokeRect(x - 10, y - 10, 20, 20);
            ctx.font = '16px monospace'; ctx.fillText(i.toString(), x + 14, y + 5);
        }
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)'; ctx.lineWidth = 1; ctx.setLineDash([8, 8]);
        ctx.beginPath(); ctx.arc(center.x, center.y, radius * 0.78, 0, 2 * Math.PI); ctx.stroke();
        ctx.setLineDash([]);
    }

    onSelect(fn) { this._onSelect = fn; }
}

// ---------------------------------------------------------------------------
// App Controller
// ---------------------------------------------------------------------------

class VideoRPMApp {
    constructor() {
        this.videoInput = document.getElementById('videoInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.videoInfo = document.getElementById('videoInfo');
        this.videoInfoText = document.getElementById('videoInfoText');
        this.flywheelSection = document.getElementById('flywheelSection');
        this.analyzeSection = document.getElementById('analyzeSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.previewCanvas = document.getElementById('previewCanvas');
        this.canvasWrapper = document.getElementById('canvasWrapper');
        this.radiusSlider = document.getElementById('radiusSlider');
        this.radiusValue = document.getElementById('radiusValue');
        this.resetSelectionBtn = document.getElementById('resetSelectionBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.rpmNumber = document.getElementById('rpmNumber');
        this.resultMeta = document.getElementById('resultMeta');
        this.rpmChart = document.getElementById('rpmChart');
        this.fftChart = document.getElementById('fftChart');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
        this.windowSelect = document.getElementById('windowSelect');
        this.minRpmInput = document.getElementById('minRpmInput');
        this.maxRpmInput = document.getElementById('maxRpmInput');
        this.hiddenVideo = document.getElementById('hiddenVideo');

        // Explainability elements
        this.explainSummary = document.getElementById('explainSummary');
        this.filmstripScroll = document.getElementById('filmstripScroll');
        this.brightnessChart = document.getElementById('brightnessChart');
        this.rotationTableBody = document.getElementById('rotationTableBody');
        this.rotationToggle = document.getElementById('rotationToggle');
        this.rotationTableWrap = document.getElementById('rotationTableWrap');

        this.videoFile = null;
        this.videoURL = null;
        this.fps = 0;
        this.selector = new FlywheelSelector(this.previewCanvas, this.canvasWrapper);

        // Version
        const vEl = document.querySelector('.version');
        if (vEl) vEl.textContent = `v${APP_VERSION}`;

        this._bindEvents();
    }

    _resetSelectionUI() {
        this.selector.center = null;
        this.selector.roiPositions = [];
        this.radiusSlider.disabled = true;
        this.resetSelectionBtn.disabled = true;
        this.analyzeBtn.disabled = true;
        this.radiusValue.textContent = '—';
    }

    _bindEvents() {
        this.videoInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this._handleFile(e.target.files[0]);
            e.target.value = '';
        });
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault(); this.uploadArea.classList.add('drag-over');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('drag-over');
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault(); this.uploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) this._handleFile(e.dataTransfer.files[0]);
        });
        this.radiusSlider.addEventListener('input', () => {
            const r = parseInt(this.radiusSlider.value, 10);
            this.radiusValue.textContent = `${r} px`;
            this.selector.setRadius(r);
        });
        this.selector.onSelect((center, radius) => {
            this.radiusSlider.disabled = false;
            this.resetSelectionBtn.disabled = false;
            this.analyzeBtn.disabled = false;
            this.radiusValue.textContent = `${radius} px`;
        });
        this.resetSelectionBtn.addEventListener('click', () => {
            this._resetSelectionUI(); this.selector.redraw();
        });
        this.analyzeBtn.addEventListener('click', () => this._runAnalysis());
        this.newAnalysisBtn.addEventListener('click', () => {
            this.resultsSection.hidden = true;
            this.flywheelSection.hidden = false;
            this.analyzeSection.hidden = false;
            this._resetSelectionUI();
            this.selector.redraw();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        // Rotation table toggle
        if (this.rotationToggle) {
            this.rotationToggle.addEventListener('click', () => {
                const wrap = this.rotationTableWrap;
                const isHidden = wrap.hidden;
                wrap.hidden = !isHidden;
                this.rotationToggle.textContent = isHidden
                    ? 'Rotation Details ▲' : 'Rotation Details ▼';
            });
        }
    }

    async _handleFile(file) {
        this.videoFile = file;
        this._resetSelectionUI();
        this.progressContainer.hidden = true;
        if (this.videoURL) URL.revokeObjectURL(this.videoURL);
        this.videoURL = URL.createObjectURL(file);
        const v = this.hiddenVideo;
        v.src = this.videoURL;
        await new Promise((res, rej) => {
            v.onloadedmetadata = res;
            v.onerror = () => rej(new Error('Cannot load video'));
        });
        if (v.readyState < 2) await new Promise(r => { v.oncanplay = r; });

        this.fps = await this._estimateFPS(v);
        const duration = v.duration;
        const sizeMB = (file.size / 1048576).toFixed(1);
        this.videoInfo.hidden = false;
        this.videoInfoText.textContent =
            `${v.videoWidth}×${v.videoHeight} · ${this.fps} fps · ${duration.toFixed(1)}s · ${sizeMB} MB`;

        await seekVideo(v, Math.min(0.5, duration * 0.1));
        await this.selector.showFrame(v);

        const defaultR = Math.round(Math.min(v.videoWidth, v.videoHeight) * 0.35);
        this.radiusSlider.value = defaultR;
        this.radiusSlider.max = Math.round(Math.min(v.videoWidth, v.videoHeight) * 0.5);
        this.selector.radius = defaultR;
        this.maxRpmInput.value = Math.min(3000, Math.floor((this.fps / 2) * 60));

        this.flywheelSection.hidden = false;
        this.analyzeSection.hidden = false;
        this.resultsSection.hidden = true;
        this.flywheelSection.scrollIntoView({ behavior: 'smooth' });
    }

    async _estimateFPS(video) {
        if ('requestVideoFrameCallback' in video) {
            return new Promise((resolve) => {
                const ts = []; let count = 0, resolved = false;
                video.muted = true; video.currentTime = 0;
                const done = (fps) => { if (resolved) return; resolved = true; video.pause(); resolve(fps); };
                const onFrame = (_, meta) => {
                    if (resolved) return;
                    ts.push(meta.mediaTime); count++;
                    if (count >= 15) {
                        const diffs = [];
                        for (let i = 1; i < ts.length; i++) { const d = ts[i] - ts[i-1]; if (d > 0) diffs.push(d); }
                        if (diffs.length > 2) {
                            diffs.sort((a, b) => a - b);
                            done(this._snapFPS(1.0 / diffs[Math.floor(diffs.length / 2)]));
                        } else done(30);
                    } else video.requestVideoFrameCallback(onFrame);
                };
                video.requestVideoFrameCallback(onFrame);
                video.play().catch(() => done(30));
                setTimeout(() => done(30), 3000);
            });
        }
        const bps = video.duration > 0 ? this.videoFile.size / video.duration : 0;
        return bps > 2_000_000 ? 60 : 30;
    }

    _snapFPS(raw) {
        const common = [24, 25, 29.97, 30, 48, 50, 59.94, 60, 120, 240];
        return common.reduce((best, f) => Math.abs(raw - f) < Math.abs(raw - best) ? f : best, 30);
    }

    async _runAnalysis() {
        const { selector, hiddenVideo, fps } = this;
        if (!selector.center || selector.roiPositions.length === 0) {
            alert('Please tap on the flywheel center first.'); return;
        }

        const windowSec = parseFloat(this.windowSelect.value);
        const minRPM = parseInt(this.minRpmInput.value, 10);
        const maxRPM = parseInt(this.maxRpmInput.value, 10);

        this.analyzeBtn.disabled = true;
        this.progressContainer.hidden = false;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = 'Extracting frames…';

        const extractCanvas = document.createElement('canvas');

        try {
            // Step 1: Extract brightness
            const signals = await extractFrames(
                hiddenVideo, extractCanvas, selector.roiPositions, fps,
                (done, total) => {
                    const pct = Math.round((done / total) * 100);
                    this.progressFill.style.width = `${pct}%`;
                    this.progressText.textContent = `Extracting frames: ${done}/${total} (${pct}%)`;
                }
            );

            // Step 2: Windowed FFT
            this.progressText.textContent = 'Computing FFT…';
            await new Promise(r => setTimeout(r, 50));
            const allTraces = [];
            let times = [];
            for (const sig of signals) {
                const result = windowedFFT(sig, fps, windowSec, 0.25, minRPM, maxRPM);
                allTraces.push(result); times = result.times;
            }
            if (times.length === 0) {
                alert('Video too short for FFT window. Reduce window size.');
                this.analyzeBtn.disabled = false; this.progressContainer.hidden = true; return;
            }

            // Step 3: Combine
            const { bestRPMs, overallRPM } = combineResults(allTraces);

            // Step 4: Find best ROI
            let bestROIIdx = 0, bestAvgConf = 0;
            for (let i = 0; i < allTraces.length; i++) {
                const avg = allTraces[i].confidences.reduce((a, b) => a + b, 0) / allTraces[i].confidences.length;
                if (avg > bestAvgConf) { bestAvgConf = avg; bestROIIdx = i; }
            }
            const bestSignal = signals[bestROIIdx];

            // Step 5: Detect peaks
            this.progressText.textContent = 'Detecting rotations…';
            await new Promise(r => setTimeout(r, 30));
            const peaks = detectPeaks(bestSignal, fps, maxRPM);
            const { rotations, peakCountRPM } = analyzeRotations(peaks, fps);

            // Step 6: Full-video FFT
            const mean = bestSignal.reduce((a, b) => a + b, 0) / bestSignal.length;
            const fullFFT = rfftMagnitude(bestSignal.map(v => v - mean), fps);

            // Step 7: Capture filmstrip
            this.progressText.textContent = 'Generating filmstrip…';
            const cx = selector.center.x, cy = selector.center.y, r = selector.radius;
            const filmstrip = await captureFilmstrip(
                hiddenVideo, peaks, fps, overallRPM,
                cx, cy, r, selector.roiPositions,
                (done, total) => {
                    this.progressText.textContent = `Filmstrip: ${done}/${total}`;
                },
                120
            );

            // Step 8: Show results
            this._showResults({
                overallRPM, bestRPMs, times, fps,
                duration: hiddenVideo.duration,
                fullFFT, nyquistRPM: (fps / 2) * 60,
                bestSignal, peaks, rotations, peakCountRPM, filmstrip,
            });

        } catch (err) {
            console.error('Analysis error:', err);
            alert('Error during analysis: ' + err.message);
        } finally {
            this.analyzeBtn.disabled = false;
            this.progressContainer.hidden = true;
        }
    }

    _showResults({ overallRPM, bestRPMs, times, fps, duration, fullFFT,
                   nyquistRPM, bestSignal, peaks, rotations, peakCountRPM, filmstrip }) {
        this.flywheelSection.hidden = true;
        this.analyzeSection.hidden = true;
        this.resultsSection.hidden = false;

        // Big RPM
        this.rpmNumber.textContent = Math.round(overallRPM);

        // Meta
        this.resultMeta.innerHTML = `
            <div class="meta-item">
                <span class="meta-label">Video FPS</span>
                <span class="meta-value">${fps}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Duration</span>
                <span class="meta-value">${duration.toFixed(1)}s</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Max Measurable</span>
                <span class="meta-value">${Math.round(nyquistRPM)} RPM</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Frequency</span>
                <span class="meta-value">${(overallRPM / 60).toFixed(1)} Hz</span>
            </div>
        `;

        // --- EXPLAINABILITY ---

        // Summary
        if (this.explainSummary) {
            const freqHz = (overallRPM / 60).toFixed(1);
            this.explainSummary.innerHTML = `
                <p>The white stripe was detected passing the observation point
                   <strong>${peaks.length} times</strong> in <strong>${duration.toFixed(1)}s</strong>.</p>
                <p>Peak counting: <strong>${peaks.length} passes</strong> →
                   <strong>${Math.round(peakCountRPM)} RPM</strong></p>
                <p>FFT analysis (more precise): <strong>${Math.round(overallRPM)} RPM</strong>
                   (${freqHz} rotations/sec)</p>
                ${Math.abs(peakCountRPM - overallRPM) > overallRPM * 0.15
                    ? '<p class="explain-note">⚠ Peak counting and FFT differ — some stripe passes may have been missed or extra peaks detected. The FFT result is more reliable.</p>'
                    : '<p class="explain-note">✓ Both methods agree — high confidence in the result.</p>'}
            `;
        }

        // Filmstrip
        if (this.filmstripScroll) {
            this.filmstripScroll.innerHTML = '';
            for (const thumb of filmstrip) {
                const item = document.createElement('div');
                item.className = 'filmstrip-item' + (thumb.isPeak ? ' filmstrip-peak' : '');
                item.appendChild(thumb.canvas);
                const label = document.createElement('span');
                label.className = 'filmstrip-label';
                label.textContent = `${thumb.time.toFixed(2)}s`;
                item.appendChild(label);
                if (thumb.isPeak) {
                    const star = document.createElement('span');
                    star.className = 'filmstrip-star';
                    star.textContent = '★';
                    item.appendChild(star);
                }
                this.filmstripScroll.appendChild(item);
            }
        }

        // Annotated brightness
        if (this.brightnessChart) {
            drawAnnotatedBrightness(this.brightnessChart, bestSignal, fps, peaks, overallRPM);
        }

        // Rotation table
        if (this.rotationTableBody) {
            const maxRows = 50;
            const rows = rotations.slice(0, maxRows);
            this.rotationTableBody.innerHTML = rows.map(r => `
                <tr>
                    <td>${r.num}</td>
                    <td>f${r.frame}</td>
                    <td>${r.time.toFixed(3)}s</td>
                    <td>${r.period !== null ? (r.period * 1000).toFixed(1) + 'ms' : '—'}</td>
                    <td>${r.rpm !== null ? Math.round(r.rpm) : '—'}</td>
                </tr>
            `).join('');
            if (rotations.length > maxRows) {
                this.rotationTableBody.innerHTML += `<tr><td colspan="5" style="text-align:center;color:#6b7280;">… ${rotations.length - maxRows} more rows</td></tr>`;
            }
            // Reset toggle state
            if (this.rotationTableWrap) this.rotationTableWrap.hidden = true;
            if (this.rotationToggle) this.rotationToggle.textContent = 'Rotation Details ▼';
        }

        // RPM over time chart
        drawLineChart(this.rpmChart, times, bestRPMs, {
            color: '#0066cc', fillColor: 'rgba(0, 102, 204, 0.08)',
            xLabel: 'Time (s)', yMin: 0,
            yMax: Math.max(overallRPM * 1.5, Math.max(...bestRPMs) * 1.2),
            highlightY: overallRPM,
            highlightLabel: `${Math.round(overallRPM)} RPM`,
        });

        // FFT spectrum (enhanced with peak annotation)
        const rpmAxis = [], specMag = [];
        for (let i = 0; i < fullFFT.freqs.length; i++) {
            const rpm = fullFFT.freqs[i] * 60;
            if (rpm >= 100 && rpm <= nyquistRPM) { rpmAxis.push(rpm); specMag.push(fullFFT.magnitudes[i]); }
        }
        drawLineChart(this.fftChart, rpmAxis, specMag, {
            color: '#e53935', fillColor: 'rgba(229, 57, 53, 0.08)',
            xLabel: 'RPM', yMin: 0,
            peakAnnotation: { freq: overallRPM / 60, rpm: overallRPM },
        });

        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => { new VideoRPMApp(); });

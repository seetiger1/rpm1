/* ==========================================================================
   Video RPM Analyzer – Client-Side JavaScript  (v1.1.0)
   Ports the Python FFT-based flywheel stripe detection to the browser.
   No server needed — runs entirely on the device.
   ========================================================================== */

'use strict';

const APP_VERSION = '1.1.0';

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
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    // Butterfly stages
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
 * Returns { freqs: Float64Array, magnitudes: Float64Array }
 */
function rfftMagnitude(signal, sampleRate) {
    const N = nextPow2(signal.length);
    const re = new Float64Array(N);
    const im = new Float64Array(N);

    // Apply Hann window and copy
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

/**
 * Compute ROI positions evenly spaced around a circle.
 */
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

/**
 * Extract mean brightness at each ROI position from ImageData.
 * @param {ImageData} imageData  Full-frame image data from canvas.
 * @param {Array} positions      Array of {x, y} ROI centers.
 * @param {number} patchHalf     Half-size of sampling patch (default 10).
 * @returns {number[]}           Brightness values per ROI.
 */
function extractBrightness(imageData, positions, patchHalf = 10) {
    const { data, width, height } = imageData;
    const values = [];

    for (const { x, y } of positions) {
        let sum = 0;
        let count = 0;
        const x0 = Math.max(0, x - patchHalf);
        const x1 = Math.min(width - 1, x + patchHalf);
        const y0 = Math.max(0, y - patchHalf);
        const y1 = Math.min(height - 1, y + patchHalf);

        for (let py = y0; py <= y1; py++) {
            for (let px = x0; px <= x1; px++) {
                const idx = (py * width + px) * 4;
                // Luminance: 0.299R + 0.587G + 0.114B
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                sum += lum;
                count++;
            }
        }
        values.push(count > 0 ? sum / count : 0);
    }
    return values;
}

/**
 * Windowed FFT to compute RPM for a single brightness signal.
 * @returns {{ times, rpms, confidences }}
 */
function windowedFFT(signal, fps, windowSec = 2.0, stepSec = 0.25,
                     minRPM = 200, maxRPM = 3000) {
    const windowFrames = Math.round(windowSec * fps);
    const stepFrames = Math.max(1, Math.round(stepSec * fps));

    // Cap by Nyquist
    const nyquistRPM = (fps / 2) * 60;
    maxRPM = Math.min(maxRPM, nyquistRPM * 0.95);

    const minFreq = minRPM / 60;
    const maxFreq = maxRPM / 60;

    const times = [];
    const rpms = [];
    const confidences = [];

    for (let i = 0; i + windowFrames <= signal.length; i += stepFrames) {
        const chunk = signal.slice(i, i + windowFrames);

        // Remove DC
        let mean = 0;
        for (let j = 0; j < chunk.length; j++) mean += chunk[j];
        mean /= chunk.length;
        for (let j = 0; j < chunk.length; j++) chunk[j] -= mean;

        const { freqs, magnitudes } = rfftMagnitude(chunk, fps);

        // Find peak in RPM range
        let peakMag = 0;
        let peakFreq = 0;
        let sumMag = 0;
        let countInRange = 0;

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
        const confidence = peakMag / (meanMag + 1e-10);

        times.push((i + windowFrames / 2) / fps);
        rpms.push(peakFreq * 60);
        confidences.push(confidence);
    }
    return { times, rpms, confidences };
}

/**
 * Combine per-ROI results: pick highest-confidence ROI per time window.
 * Returns { bestRPMs, overallRPM }.
 */
function combineResults(allTraces) {
    // allTraces: array of { rpms, confidences } per ROI
    const nWindows = allTraces[0].rpms.length;
    const bestRPMs = new Float64Array(nWindows);

    for (let t = 0; t < nWindows; t++) {
        let bestConf = -1;
        let bestRPM = 0;
        for (const trace of allTraces) {
            if (trace.confidences[t] > bestConf) {
                bestConf = trace.confidences[t];
                bestRPM = trace.rpms[t];
            }
        }
        bestRPMs[t] = bestRPM;
    }

    // Median with IQR outlier rejection
    const sorted = Array.from(bestRPMs).sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    let filtered;
    if (iqr > 0) {
        filtered = sorted.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
    } else {
        filtered = sorted;
    }
    const overallRPM = filtered.length > 0
        ? filtered[Math.floor(filtered.length / 2)]
        : sorted[Math.floor(sorted.length / 2)];

    return { bestRPMs: Array.from(bestRPMs), overallRPM };
}

// ---------------------------------------------------------------------------
// Video Frame Extraction
// ---------------------------------------------------------------------------

/**
 * Reliably seek a video element to a specific time.
 * Returns a promise that resolves when the frame is ready to draw.
 */
function seekVideo(video, timeSec) {
    return new Promise((resolve) => {
        // If already at target time (within one frame), resolve immediately
        if (Math.abs(video.currentTime - timeSec) < 0.001) {
            resolve();
            return;
        }

        const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = timeSec;

        // Safety timeout — some browsers don't fire seeked at end of video
        setTimeout(() => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        }, 500);
    });
}

/**
 * Seek-based frame-by-frame extraction from a video element.
 * Draws each frame to a canvas and extracts brightness at ROI points.
 */
async function extractFrames(video, canvas, roiPositions, fps, onProgress) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const duration = video.duration;
    const dt = 1.0 / fps;
    const totalFrames = Math.floor(duration * fps);

    // We'll collect brightness for each ROI
    const nROIs = roiPositions.length;
    const signals = Array.from({ length: nROIs }, () => []);

    let extractedCount = 0;

    for (let t = 0; t < duration - dt / 2; t += dt) {
        await seekVideo(video, t);

        ctx.drawImage(video, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const brightness = extractBrightness(imageData, roiPositions);

        for (let j = 0; j < nROIs; j++) {
            signals[j].push(brightness[j]);
        }

        extractedCount++;
        if (onProgress) {
            onProgress(extractedCount, totalFrames);
        }

        // Yield to UI every 20 frames to keep progress bar responsive
        if (extractedCount % 20 === 0) {
            await new Promise((r) => setTimeout(r, 0));
        }
    }

    return signals;
}

// ---------------------------------------------------------------------------
// Chart Drawing (lightweight canvas-based)
// ---------------------------------------------------------------------------

function drawLineChart(canvas, xData, yData, {
    color = '#0066cc',
    fillColor = 'rgba(0, 102, 204, 0.1)',
    lineWidth = 2,
    xLabel = '',
    yLabel = '',
    yMin = null,
    yMax = null,
    highlightY = null,
    highlightLabel = '',
} = {}) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 12, right: 12, bottom: 28, left: 48 };

    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Compute ranges
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    if (yMin === null) yMin = Math.min(...yData) * 0.9;
    if (yMax === null) yMax = Math.max(...yData) * 1.1;
    if (yMax <= yMin) yMax = yMin + 1;

    const toX = (v) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const toY = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    const nGridY = 4;
    for (let i = 0; i <= nGridY; i++) {
        const yVal = yMin + (yMax - yMin) * i / nGridY;
        const py = toY(yVal);
        ctx.beginPath();
        ctx.moveTo(pad.left, py);
        ctx.lineTo(W - pad.right, py);
        ctx.stroke();

        // Y-axis label
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(yVal).toString(), pad.left - 6, py + 3);
    }

    // Highlight line (e.g. median RPM)
    if (highlightY !== null) {
        const hy = toY(highlightY);
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(pad.left, hy);
        ctx.lineTo(W - pad.right, hy);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#4caf50';
        ctx.font = 'bold 11px -apple-system, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(highlightLabel, pad.left + 4, hy - 4);
    }

    // Data line
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < xData.length; i++) {
        const px = toX(xData[i]);
        const py = toY(yData[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Fill under line
    if (fillColor) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(toX(xData[0]), toY(yData[0]));
        for (let i = 1; i < xData.length; i++) {
            ctx.lineTo(toX(xData[i]), toY(yData[i]));
        }
        ctx.lineTo(toX(xData[xData.length - 1]), toY(yMin));
        ctx.lineTo(toX(xData[0]), toY(yMin));
        ctx.closePath();
        ctx.fill();
    }

    // X-axis label
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    const nLabelsX = Math.min(6, xData.length);
    for (let i = 0; i < nLabelsX; i++) {
        const idx = Math.round(i * (xData.length - 1) / (nLabelsX - 1));
        ctx.fillText(xData[idx].toFixed(1), toX(xData[idx]), H - 4);
    }

    // Axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    if (xLabel) ctx.fillText(xLabel, W / 2, H - 2);
}

// ---------------------------------------------------------------------------
// Canvas Flywheel Selection UI
// ---------------------------------------------------------------------------

class FlywheelSelector {
    constructor(canvasEl, wrapperEl) {
        this.canvas = canvasEl;
        this.wrapper = wrapperEl;
        this.ctx = canvasEl.getContext('2d');
        this.frameImage = null; // Stores a snapshot (ImageBitmap or ImageData)
        this.center = null;     // { x, y } in image coordinates
        this.radius = 200;      // in image coordinates
        this.roiPositions = []; // computed ROI positions
        this.imageW = 0;
        this.imageH = 0;

        this._onSelect = null;

        // Touch/mouse events on the wrapper
        this.wrapper.addEventListener('pointerdown', (e) => this._handleTap(e));
    }

    /**
     * Capture a snapshot of the current video frame and display it.
     * We store the snapshot as an ImageBitmap so it remains drawable
     * even after the video element seeks to a different time later.
     */
    async showFrame(video) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        if (vw === 0 || vh === 0) return;

        this.imageW = vw;
        this.imageH = vh;

        // Set canvas to native video resolution
        this.canvas.width = vw;
        this.canvas.height = vh;

        // Create a persistent snapshot of this frame.
        // createImageBitmap works in all modern browsers (Safari 15+, Chrome, Firefox).
        try {
            this.frameImage = await createImageBitmap(video);
        } catch (_err) {
            // Fallback: draw to a temp canvas and capture as ImageData
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = vw;
            tmpCanvas.height = vh;
            const tmpCtx = tmpCanvas.getContext('2d');
            tmpCtx.drawImage(video, 0, 0);
            this.frameImage = tmpCanvas; // store the canvas itself as a drawable
        }

        this.center = null;
        this.roiPositions = [];
        this.ctx.drawImage(this.frameImage, 0, 0);
    }

    /** Handle a tap to set the flywheel center. */
    _handleTap(e) {
        e.preventDefault();
        if (!this.frameImage) return;

        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const scaleX = this.imageW / rect.width;
        const scaleY = this.imageH / rect.height;

        const displayX = e.clientX - rect.left;
        const displayY = e.clientY - rect.top;

        // Convert display coordinates to image coordinates
        const imgX = Math.round(displayX * scaleX);
        const imgY = Math.round(displayY * scaleY);

        // Sanity check — must be within the image
        if (imgX < 0 || imgX >= this.imageW || imgY < 0 || imgY >= this.imageH) return;

        this.center = { x: imgX, y: imgY };
        this._updateROIs();
        this.redraw();

        if (this._onSelect) this._onSelect(this.center, this.radius);
    }

    /** Update radius (called from slider). */
    setRadius(r) {
        this.radius = r;
        if (this.center) {
            this._updateROIs();
            this.redraw();
        }
    }

    /** Compute ROI positions based on current center + radius. */
    _updateROIs() {
        if (!this.center) return;
        this.roiPositions = computeROIPositions(
            this.center.x, this.center.y, this.radius, 8, 0.78
        );
    }

    /** Redraw the snapshot frame with overlay. */
    redraw() {
        const { ctx, frameImage, center, radius } = this;
        if (!frameImage) return;

        ctx.drawImage(frameImage, 0, 0);

        if (!center) return;

        // Draw flywheel circle
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);
        ctx.stroke();

        // Draw center dot
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(center.x, center.y, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Draw ROI positions
        ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.9)';
        ctx.lineWidth = 2;
        const ps = 10;
        for (let i = 0; i < this.roiPositions.length; i++) {
            const { x, y } = this.roiPositions[i];
            ctx.strokeRect(x - ps, y - ps, ps * 2, ps * 2);
            ctx.font = '16px monospace';
            ctx.fillText(i.toString(), x + ps + 4, y + 5);
        }

        // Draw inner ring (sampling ring)
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([8, 8]);
        ctx.beginPath();
        ctx.arc(center.x, center.y, radius * 0.78, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    /** Register callback when selection changes. */
    onSelect(fn) { this._onSelect = fn; }
}

// ---------------------------------------------------------------------------
// App Controller
// ---------------------------------------------------------------------------

class VideoRPMApp {
    constructor() {
        // DOM elements
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

        // State
        this.videoFile = null;
        this.videoURL = null;
        this.fps = 0;
        this.selector = new FlywheelSelector(this.previewCanvas, this.canvasWrapper);

        // Show version in footer
        const footer = document.querySelector('.app-footer');
        if (footer) {
            const versionEl = footer.querySelector('.version');
            if (versionEl) versionEl.textContent = `v${APP_VERSION}`;
        }

        this._bindEvents();
    }

    _resetSelectionUI() {
        this.selector.center = null;
        this.selector.roiPositions = [];;
        this.radiusSlider.disabled = true;
        this.resetSelectionBtn.disabled = true;
        this.analyzeBtn.disabled = true;
        this.radiusValue.textContent = '—';
    }

    _bindEvents() {
        // File upload
        this.videoInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this._handleFile(e.target.files[0]);
            }
            // Reset input so the same file can be re-selected
            e.target.value = '';
        });

        // Drag & drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('drag-over');
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) this._handleFile(e.dataTransfer.files[0]);
        });

        // Radius slider
        this.radiusSlider.addEventListener('input', () => {
            const r = parseInt(this.radiusSlider.value, 10);
            this.radiusValue.textContent = `${r} px`;
            this.selector.setRadius(r);
        });

        // Flywheel selection callback
        this.selector.onSelect((center, radius) => {
            this.radiusSlider.disabled = false;
            this.resetSelectionBtn.disabled = false;
            this.analyzeBtn.disabled = false;
            this.radiusValue.textContent = `${radius} px`;
        });

        // Reset selection
        this.resetSelectionBtn.addEventListener('click', () => {
            this._resetSelectionUI();
            this.selector.redraw();
        });

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this._runAnalysis());

        // New analysis — go back to flywheel selection with same video
        this.newAnalysisBtn.addEventListener('click', () => {
            this.resultsSection.hidden = true;
            this.flywheelSection.hidden = false;
            this.analyzeSection.hidden = false;

            // Reset selection state but keep the frame snapshot
            this._resetSelectionUI();
            this.selector.redraw(); // redraws the stored snapshot without overlay

            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    async _handleFile(file) {
        this.videoFile = file;

        // Reset UI
        this._resetSelectionUI();
        this.progressContainer.hidden = true;

        // Revoke previous blob URL to free memory
        if (this.videoURL) {
            URL.revokeObjectURL(this.videoURL);
        }
        this.videoURL = URL.createObjectURL(file);

        // Load the video element
        const v = this.hiddenVideo;
        v.src = this.videoURL;

        await new Promise((resolve, reject) => {
            v.onloadedmetadata = resolve;
            v.onerror = () => reject(new Error('Cannot load video'));
        });

        // Wait for enough data to seek
        if (v.readyState < 2) {
            await new Promise((resolve) => {
                v.oncanplay = resolve;
            });
        }

        // Estimate FPS
        this.fps = await this._estimateFPS(v);

        const duration = v.duration;
        const sizeMB = (file.size / 1048576).toFixed(1);

        // Show info
        this.videoInfo.hidden = false;
        this.videoInfoText.textContent =
            `${v.videoWidth}×${v.videoHeight} · ${this.fps} fps · ${duration.toFixed(1)}s · ${sizeMB} MB`;

        // Seek to 0.5s (or 0 for very short clips) and snapshot the frame
        const seekTarget = Math.min(0.5, duration * 0.1);
        await seekVideo(v, seekTarget);

        // showFrame creates a persistent ImageBitmap snapshot
        await this.selector.showFrame(v);

        // Set default radius based on frame size
        const defaultRadius = Math.round(Math.min(v.videoWidth, v.videoHeight) * 0.35);
        this.radiusSlider.value = defaultRadius;
        this.radiusSlider.max = Math.round(Math.min(v.videoWidth, v.videoHeight) * 0.5);
        this.selector.radius = defaultRadius;

        // Calculate Nyquist limit
        const nyquistRPM = Math.floor((this.fps / 2) * 60);
        this.maxRpmInput.value = Math.min(3000, nyquistRPM);

        // Show sections
        this.flywheelSection.hidden = false;
        this.analyzeSection.hidden = false;
        this.resultsSection.hidden = true;

        // Scroll to flywheel section
        this.flywheelSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Estimate video FPS by measuring frame timestamps.
     * Uses requestVideoFrameCallback if available, else falls back to heuristic.
     */
    async _estimateFPS(video) {
        // Method 1: requestVideoFrameCallback (modern browsers)
        if ('requestVideoFrameCallback' in video) {
            return new Promise((resolve) => {
                const timestamps = [];
                let count = 0;
                let resolved = false;

                video.muted = true;
                video.currentTime = 0;

                const done = (fps) => {
                    if (resolved) return;
                    resolved = true;
                    video.pause();
                    resolve(fps);
                };

                const onFrame = (_now, metadata) => {
                    if (resolved) return;
                    timestamps.push(metadata.mediaTime);
                    count++;
                    if (count >= 15) {
                        // Calculate average FPS from timestamps
                        const diffs = [];
                        for (let i = 1; i < timestamps.length; i++) {
                            const d = timestamps[i] - timestamps[i - 1];
                            if (d > 0) diffs.push(d);
                        }
                        if (diffs.length > 2) {
                            diffs.sort((a, b) => a - b);
                            const medianDt = diffs[Math.floor(diffs.length / 2)];
                            const rawFPS = 1.0 / medianDt;
                            done(this._snapFPS(rawFPS));
                        } else {
                            done(30);
                        }
                    } else {
                        video.requestVideoFrameCallback(onFrame);
                    }
                };

                video.requestVideoFrameCallback(onFrame);
                video.play().catch(() => done(30));

                // Timeout fallback
                setTimeout(() => done(30), 3000);
            });
        }

        // Method 2: Heuristic based on file size and duration
        const bytesPerSec = video.duration > 0 ? this.videoFile.size / video.duration : 0;
        return bytesPerSec > 2_000_000 ? 60 : 30;
    }

    _snapFPS(rawFPS) {
        const common = [24, 25, 29.97, 30, 48, 50, 59.94, 60, 120, 240];
        let best = 30;
        let bestDiff = Infinity;
        for (const fps of common) {
            const diff = Math.abs(rawFPS - fps);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = fps;
            }
        }
        return best;
    }

    async _runAnalysis() {
        const { selector, hiddenVideo, fps } = this;

        if (!selector.center || selector.roiPositions.length === 0) {
            alert('Please tap on the flywheel center first.');
            return;
        }

        const windowSec = parseFloat(this.windowSelect.value);
        const minRPM = parseInt(this.minRpmInput.value, 10);
        const maxRPM = parseInt(this.maxRpmInput.value, 10);

        // UI: show progress
        this.analyzeBtn.disabled = true;
        this.progressContainer.hidden = false;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = 'Extracting frames…';

        // Create an off-screen canvas for extraction (don't disturb preview)
        const extractCanvas = document.createElement('canvas');

        try {
            // Step 1: Extract brightness signals from all frames
            const signals = await extractFrames(
                hiddenVideo,
                extractCanvas,
                selector.roiPositions,
                fps,
                (done, total) => {
                    const pct = Math.round((done / total) * 100);
                    this.progressFill.style.width = `${pct}%`;
                    this.progressText.textContent =
                        `Extracting frames: ${done}/${total} (${pct}%)`;
                }
            );

            // Step 2: Windowed FFT on each ROI
            this.progressText.textContent = 'Computing FFT…';
            this.progressFill.style.width = '100%';

            // Small delay to let the UI update
            await new Promise((r) => setTimeout(r, 50));

            const allTraces = [];
            let times = [];

            for (const sig of signals) {
                const result = windowedFFT(sig, fps, windowSec, 0.25, minRPM, maxRPM);
                allTraces.push(result);
                times = result.times;
            }

            if (times.length === 0) {
                alert('Video too short for the selected FFT window. Reduce window size.');
                this.analyzeBtn.disabled = false;
                this.progressContainer.hidden = true;
                return;
            }

            // Step 3: Combine results
            const { bestRPMs, overallRPM } = combineResults(allTraces);

            // Step 4: Compute full-video FFT for spectrum plot
            // Use the signal from the ROI with highest average confidence
            let bestROIIdx = 0;
            let bestAvgConf = 0;
            for (let i = 0; i < allTraces.length; i++) {
                const avg = allTraces[i].confidences.reduce((a, b) => a + b, 0) /
                            allTraces[i].confidences.length;
                if (avg > bestAvgConf) {
                    bestAvgConf = avg;
                    bestROIIdx = i;
                }
            }
            const bestSignal = signals[bestROIIdx];
            const mean = bestSignal.reduce((a, b) => a + b, 0) / bestSignal.length;
            const fullFFT = rfftMagnitude(
                bestSignal.map((v) => v - mean),
                fps
            );

            // Step 5: Display results
            this._showResults({
                overallRPM,
                bestRPMs,
                times,
                fps,
                duration: hiddenVideo.duration,
                fullFFT,
                minRPM,
                maxRPM,
                nyquistRPM: (fps / 2) * 60,
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
                   minRPM, maxRPM, nyquistRPM }) {
        // Hide input sections, show results
        this.flywheelSection.hidden = true;
        this.analyzeSection.hidden = true;
        this.resultsSection.hidden = false;

        // Big RPM number
        this.rpmNumber.textContent = Math.round(overallRPM);

        // Meta info
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

        // RPM over time chart
        drawLineChart(this.rpmChart, times, bestRPMs, {
            color: '#0066cc',
            fillColor: 'rgba(0, 102, 204, 0.08)',
            xLabel: 'Time (s)',
            yMin: 0,
            yMax: Math.max(overallRPM * 1.5, Math.max(...bestRPMs) * 1.2),
            highlightY: overallRPM,
            highlightLabel: `${Math.round(overallRPM)} RPM`,
        });

        // FFT spectrum chart (in RPM units)
        const rpmAxis = [];
        const specMag = [];
        for (let i = 0; i < fullFFT.freqs.length; i++) {
            const rpm = fullFFT.freqs[i] * 60;
            if (rpm >= 100 && rpm <= nyquistRPM) {
                rpmAxis.push(rpm);
                specMag.push(fullFFT.magnitudes[i]);
            }
        }

        drawLineChart(this.fftChart, rpmAxis, specMag, {
            color: '#e53935',
            fillColor: 'rgba(229, 57, 53, 0.08)',
            xLabel: 'RPM',
            yMin: 0,
            highlightY: null,
        });

        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// ---------------------------------------------------------------------------
// Initialize on DOM ready
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    new VideoRPMApp();
});

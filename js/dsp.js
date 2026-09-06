/**
 * Digital Signal Processing Module
 * Provides algorithms for frequency detection and data processing.
 *
 * Key design decision: HPS uses ADDITIVE approach in dB domain
 * (equivalent to multiplication in linear domain) to avoid
 * numerical underflow when multiplying very small linear magnitudes.
 */

export const RPM_RANGES = {
  idle: {
    id: 'idle',
    name: 'Standgas',
    label: '400–800 RPM',
    minRPM: 400,
    maxRPM: 800,
    numHarmonics: 7,
    power: 0.85,
    filterCutoff: 150
  },
  cruise: {
    id: 'cruise',
    name: 'Fahrt',
    label: '800–2000 RPM',
    minRPM: 800,
    maxRPM: 2000,
    numHarmonics: 4,
    power: 0.80,
    filterCutoff: 250
  },
  high: {
    id: 'high',
    name: 'Volllast',
    label: '1800–3500 RPM',
    minRPM: 1800,
    maxRPM: 3500,
    numHarmonics: 3,
    power: 0.70,
    filterCutoff: 380
  },
  auto: {
    id: 'auto',
    name: 'Auto',
    label: '400–3500 RPM',
    minRPM: 400,
    maxRPM: 3500,
    numHarmonics: 4,
    power: 0.82,
    filterCutoff: 250
  }
};

/**
 * Implements the Harmonic Product Spectrum (HPS) algorithm in dB domain.
 * Uses addition of dB values (equivalent to multiplication of linear magnitudes)
 * to avoid numerical underflow issues.
 *
 * @param {Float32Array} spectrum - FFT frequency data in dB from AnalyserNode.getFloatFrequencyData()
 * @param {number} [numHarmonics=4] - Number of harmonics to combine
 * @param {number} [power=0.82] - Frequency weighting power
 * @returns {Float32Array} The resulting HPS array (in dB, higher = stronger fundamental)
 */
export function harmonicProductSpectrum(spectrum, numHarmonics = 4, power = 0.82) {
  const length = spectrum.length;
  const hps = new Float32Array(length);

  // Convert dB to linear magnitude (Harmonic Sum Spectrum)
  for (let i = 0; i < length; i++) {
    hps[i] = Math.pow(10, spectrum[i] / 20);
  }

  // Add downsampled spectra
  for (let h = 2; h <= numHarmonics; h++) {
    for (let i = 0; i < length; i++) {
      const idx = i * h;
      if (idx < length) {
        hps[i] += Math.pow(10, spectrum[idx] / 20);
      }
    }
  }

  // Balanced frequency weighting
  for (let i = 1; i < length; i++) {
    hps[i] = hps[i] / Math.pow(i, power);
  }

  // Convert back to pseudo-dB for peak finding thresholds
  for (let i = 0; i < length; i++) {
    hps[i] = 20 * Math.log10(hps[i] + 1e-10);
  }

  return hps;
}

/**
 * Finds the dominant frequency peak in the HPS spectrum within a given range.
 * Uses both an absolute noise floor and a relative peak-to-median check.
 *
 * @param {Float32Array} hpsSpectrum - The HPS array (in dB, from harmonicProductSpectrum)
 * @param {number} sampleRate - The sample rate of the audio context
 * @param {number} fftSize - The FFT size used by the AnalyserNode
 * @param {number} [minFreq=15] - Minimum frequency to search (Hz)
 * @param {number} [maxFreq=500] - Maximum frequency to search (Hz)
 * @returns {{ frequency: number, binIndex: number, magnitude: number }}
 */
export function findPeakFrequency(hpsSpectrum, sampleRate, fftSize, minFreq = 15, maxFreq = 500) {
  const binCount = hpsSpectrum.length; // frequencyBinCount = fftSize / 2
  const freqPerBin = (sampleRate / 2) / binCount;

  const minBin = Math.max(1, Math.ceil(minFreq / freqPerBin));
  const maxBin = Math.min(binCount - 1, Math.ceil(maxFreq / freqPerBin));

  // Find peak and compute mean in the search range
  let maxVal = -Infinity;
  let maxIndex = -1;
  let sum = 0;
  let count = 0;

  for (let i = minBin; i <= maxBin; i++) {
    const val = hpsSpectrum[i];
    sum += val;
    count++;
    if (val > maxVal) {
      maxVal = val;
      maxIndex = i;
    }
  }

  // Noise gate for HSS: silence is around -100 to -150 dB.
  const ABSOLUTE_FLOOR_DB = -120;
  if (maxVal < ABSOLUTE_FLOOR_DB || maxIndex === -1) {
    return { frequency: 0, binIndex: -1, magnitude: 0 };
  }

  // Relative check: peak should stand out above the mean
  const mean = sum / count;
  const RELATIVE_THRESHOLD_DB = 6;
  if (maxVal - mean < RELATIVE_THRESHOLD_DB) {
    return { frequency: 0, binIndex: -1, magnitude: 0 };
  }

  // Sub-bin precision
  let peakFreq = maxIndex * freqPerBin;
  if (maxIndex > minBin && maxIndex < maxBin) {
    const alpha = hpsSpectrum[maxIndex - 1];
    const beta  = hpsSpectrum[maxIndex];
    const gamma = hpsSpectrum[maxIndex + 1];
    const denom = alpha - 2 * beta + gamma;
    if (denom !== 0) {
      let p = 0.5 * (alpha - gamma) / denom;
      p = Math.max(-0.10, Math.min(0.10, p));
      peakFreq = (maxIndex + p) * freqPerBin;
    }
  }

  return { frequency: peakFreq, binIndex: maxIndex, magnitude: maxVal };
}

/**
 * Converts frequency in Hz to RPM for a 4-stroke engine.
 *
 * @param {number} frequency - Detected firing frequency in Hz
 * @param {number} [cylinders=4] - Number of engine cylinders
 * @returns {number} RPM rounded to nearest integer
 */
export function frequencyToRPM(frequency, cylinders = 4) {
  if (frequency <= 0 || cylinders <= 0) return 0;
  // For 4-stroke: RPM = (Hz × 120) / cylinders
  return Math.round((frequency * 120) / cylinders);
}

/**
 * Moving average filter with circular buffer.
 * Ignores zero values to avoid pulling the average down during silence.
 */
export class MovingAverage {
  /**
   * @param {number} [windowSize=5] - Number of samples to average
   */
  constructor(windowSize = 5) {
    this.windowSize = windowSize;
    this.buffer = new Float32Array(windowSize);
    this.index = 0;
    this.count = 0;
  }

  /**
   * Adds a value and returns the current smoothed average.
   * @param {number} value - Value to add (0 values are ignored)
   * @returns {number} Current average
   */
  push(value) {
    if (value === 0) {
      return this.getAverage();
    }

    this.buffer[this.index] = value;
    this.index = (this.index + 1) % this.windowSize;
    if (this.count < this.windowSize) {
      this.count++;
    }

    return this.getAverage();
  }

  /** @returns {number} Current average, or 0 if buffer is empty */
  getAverage() {
    if (this.count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.count; i++) {
      sum += this.buffer[i];
    }
    return sum / this.count;
  }

  /** Clears the buffer */
  reset() {
    this.buffer.fill(0);
    this.index = 0;
    this.count = 0;
  }
}

/**
 * Rolling median filter to completely eliminate transient harmonic jumps.
 */
export class MovingMedian {
  /**
   * @param {number} [windowSize=7] - Number of samples to compute median
   */
  constructor(windowSize = 7) {
    this.windowSize = windowSize;
    this.buffer = [];
  }

  /**
   * Adds a value and returns the current smoothed median.
   * @param {number} value - Value to add (0 values are ignored)
   * @returns {number} Current median
   */
  push(value) {
    if (value === 0) {
      return this.getMedian();
    }

    this.buffer.push(value);
    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    return this.getMedian();
  }

  /** @returns {number} Current median, or 0 if buffer is empty */
  getMedian() {
    if (this.buffer.length === 0) return 0;
    const sorted = [...this.buffer].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 !== 0) {
      return sorted[mid];
    }
    return Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  }

  /** Clears the buffer */
  reset() {
    this.buffer = [];
  }
}

/**
 * Stores timestamped RPM values for timeline graph and CSV export.
 */
export class RPMHistory {
  constructor() {
    this.history = [];
    this.startTime = null;
  }

  /**
   * Records an RPM value with a timestamp.
   * @param {number} rpm - The RPM value
   * @param {number} timestamp - Absolute timestamp in seconds (e.g. audioContext.currentTime)
   */
  record(rpm, timestamp) {
    if (this.startTime === null) {
      this.startTime = timestamp;
    }
    const relativeTime = timestamp - this.startTime;
    this.history.push({ time: relativeTime, rpm });
  }

  /** @returns {Array<{time: number, rpm: number}>} */
  getData() {
    return this.history;
  }

  /** Resets the history */
  clear() {
    this.history = [];
    this.startTime = null;
  }

  /**
   * Generates a CSV string.
   * @returns {string} CSV with header "time_s,rpm\n"
   */
  getCSV() {
    let csv = "time_s,rpm\n";
    for (const record of this.history) {
      csv += `${record.time.toFixed(3)},${record.rpm}\n`;
    }
    return csv;
  }
}

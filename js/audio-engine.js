/**
 * Audio Engine Module
 * Handles audio input from microphone or file, routing, filtering, and analysis.
 */

export class AudioEngine {
  constructor() {
    this.mode = null; // 'file' | 'mic' | null
    this.isRunning = false;
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.bandpassFilter = null;
    this._sampleRate = 0;
    this._fftSize = 0;
    this.audioBuffer = null;
    this.playbackStartTime = 0;
    this.mediaStream = null;
  }

  /**
   * Initializes the AudioContext and audio node chain.
   * Safe to call inside a user interaction handler (required for iOS Safari).
   */
  async init() {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      
      this.bandpassFilter = this.audioContext.createBiquadFilter();
      this.bandpassFilter.type = 'lowpass';
      this.bandpassFilter.frequency.value = 150;
      this.bandpassFilter.Q.value = 0.5;

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 8192;
      this.analyser.smoothingTimeConstant = 0.8;

      this.bandpassFilter.connect(this.analyser);

      // iOS Safari workaround
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      this._sampleRate = this.audioContext.sampleRate;
      this._fftSize = this.analyser.fftSize;
    } catch (error) {
      console.error("Failed to initialize AudioEngine:", error);
      throw error;
    }
  }

  /**
   * Sets the lowpass filter cutoff frequency.
   * @param {number} freq - Frequency in Hz
   */
  setFilterCutoff(freq) {
    if (this.bandpassFilter && this.bandpassFilter.frequency) {
      this.bandpassFilter.frequency.value = freq;
    }
  }

  /**
   * Loads an audio file for processing.
   * 
   * @param {File} file - The audio file from input[type=file]
   * @returns {Promise<{duration: number}>} Object containing file duration in seconds
   */
  async loadFile(file) {
    if (!this.audioContext) {
      throw new Error("AudioEngine not initialized. Call init() first.");
    }
    
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (event) => {
        try {
          const arrayBuffer = event.target.result;
          this.audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
          
          this.mode = 'file';
          resolve({ duration: this.audioBuffer.duration });
        } catch (error) {
          console.error("Error decoding audio data:", error);
          reject(error);
        }
      };
      
      reader.onerror = (error) => {
        console.error("Error reading file:", error);
        reject(error);
      };
      
      reader.readAsArrayBuffer(file);
    });
  }

  /**
   * Starts playback of the loaded audio file.
   * 
   * @returns {Promise<void>} Resolves when playback ends
   */
  playFile() {
    if (!this.audioBuffer) {
      return Promise.reject(new Error("No audio file loaded."));
    }
    
    this.stopFile(); // Clean up existing source if any
    
    return new Promise((resolve) => {
      this.source = this.audioContext.createBufferSource();
      this.source.buffer = this.audioBuffer;
      this.source.connect(this.bandpassFilter);
      
      this.source.onended = () => {
        this.isRunning = false;
        resolve();
      };
      
      this.playbackStartTime = this.audioContext.currentTime;
      this.source.start(0);
      this.isRunning = true;
    });
  }

  /**
   * Stops playback of the audio file.
   */
  stopFile() {
    if (this.source && this.mode === 'file') {
      try {
        this.source.stop();
        this.source.disconnect();
      } catch (e) {
        // Source might already be stopped
      }
      this.source = null;
    }
    this.isRunning = false;
  }

  /**
   * Requests microphone access and starts processing microphone audio.
   * 
   * @returns {Promise<void>}
   */
  async startMicrophone() {
    if (!this.audioContext) {
      throw new Error("AudioEngine not initialized. Call init() first.");
    }

    try {
      this.stopFile(); // Stop file if running
      
      const constraints = {
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: 44100
        }
      };
      
      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      this.source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.source.connect(this.bandpassFilter);
      
      this.mode = 'mic';
      this.isRunning = true;
    } catch (error) {
      console.error("Failed to start microphone:", error);
      throw error;
    }
  }

  /**
   * Stops processing microphone audio and releases the stream.
   */
  stopMicrophone() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.source && this.mode === 'mic') {
      this.source.disconnect();
      this.source = null;
    }
    this.isRunning = false;
  }

  /**
   * Gets the current frequency data (FFT) in dB.
   * 
   * @returns {Float32Array} Array of frequency magnitude values in dB
   */
  getFrequencyData() {
    if (!this.analyser) return new Float32Array(0);
    const dataArray = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatFrequencyData(dataArray);
    return dataArray;
  }

  /**
   * Gets the current time domain data (waveform).
   * 
   * @returns {Float32Array} Array of time domain values
   */
  getTimeDomainData() {
    if (!this.analyser) return new Float32Array(0);
    const dataArray = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatTimeDomainData(dataArray);
    return dataArray;
  }

  /**
   * Cleans up all audio resources.
   */
  dispose() {
    this.stopMicrophone();
    this.stopFile();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.analyser = null;
    this.bandpassFilter = null;
  }

  // --- Public Properties ---

  /**
   * @returns {number} The current sample rate
   */
  get sampleRate() {
    return this._sampleRate;
  }

  /**
   * @returns {number} The current FFT size
   */
  get fftSize() {
    return this._fftSize;
  }

  /**
   * @returns {number} The duration of the loaded file in seconds, or 0 if no file
   */
  get duration() {
    return this.audioBuffer ? this.audioBuffer.duration : 0;
  }

  /**
   * @returns {number} Current playback time for file mode in seconds
   */
  get currentTime() {
    if (this.mode === 'file' && this.isRunning && this.audioContext) {
      return this.audioContext.currentTime - this.playbackStartTime;
    }
    return 0;
  }
}

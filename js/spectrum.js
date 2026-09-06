export class SpectrumAnalyzer {
  /**
   * Initialize SpectrumAnalyzer
   * @param {string} canvasId - The ID of the canvas element
   */
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`Canvas with ID ${canvasId} not found`);
    
    this.ctx = this.canvas.getContext('2d');
    
    this.maxDisplayFreq = 1000; // Hz
    this.isVisible = true;
    
    this.resize = this.resize.bind(this);
    
    // Resize observer
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas.parentElement);
    
    this.resize();
  }
  
  /**
   * Set visibility of the spectrum analyzer container
   * @param {boolean} visible 
   */
  setVisible(visible) {
    this.isVisible = visible;
    this.canvas.parentElement.style.display = visible ? 'block' : 'none';
    if (visible) {
      this.resize();
    }
  }
  
  /**
   * Handle responsive resize and devicePixelRatio
   */
  resize() {
    if (!this.isVisible) return;
    
    const parent = this.canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    
    const dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    
    this.ctx.scale(dpr, dpr);
    
    this.width = rect.width;
    this.height = rect.height;
  }
  
  /**
   * Draw the spectrum frame
   * @param {Uint8Array|Float32Array} frequencyData 
   * @param {number} sampleRate 
   * @param {number} fftSize 
   * @param {number|null} peakFrequency 
   */
  draw(frequencyData, sampleRate, fftSize, peakFrequency = null) {
    if (!this.isVisible) return;
    
    const ctx = this.ctx;
    
    // Clear background
    ctx.fillStyle = '#f5f5f7';
    ctx.fillRect(0, 0, this.width, this.height);
    
    const binSize = sampleRate / fftSize;
    // How many bins we need to display up to maxDisplayFreq
    const numBins = Math.min(frequencyData.length, Math.floor(this.maxDisplayFreq / binSize));
    
    const padding = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartWidth = this.width - padding.left - padding.right;
    const chartHeight = this.height - padding.top - padding.bottom;
    
    // Draw Grid and Y-axis
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#e0e0e0';
    ctx.fillStyle = '#1a1a2e';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    const numYLines = 4;
    for (let i = 0; i <= numYLines; i++) {
      const y = padding.top + (chartHeight / numYLines) * i;
      
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
      
      const val = 100 - (i * 25); // Just a generic % or 0-255 representation
      ctx.fillText(`${val}`, padding.left - 5, y);
    }
    
    // Draw X-axis
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let freq = 0; freq <= this.maxDisplayFreq; freq += 100) {
      const x = padding.left + (freq / this.maxDisplayFreq) * chartWidth;
      
      ctx.beginPath();
      ctx.moveTo(x, padding.top + chartHeight);
      ctx.lineTo(x, padding.top + chartHeight + 5);
      ctx.stroke();
      
      ctx.fillText(`${freq}`, x, padding.top + chartHeight + 8);
    }
    
    // Draw Bars
    const barWidth = Math.max(1, chartWidth / numBins);
    
    // Pre-create gradient for performance
    const barGradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartHeight);
    barGradient.addColorStop(0, '#0066cc');
    barGradient.addColorStop(1, '#80b3e6');
    ctx.fillStyle = barGradient;
    
    for (let i = 0; i < numBins; i++) {
      // getFloatFrequencyData returns dB values, typically -100 to 0
      const dbValue = frequencyData[i];
      // Map dB range [-100, 0] to [0, 1]
      const normalizedValue = Math.max(0, Math.min(1, (dbValue + 100) / 100));
      
      const height = normalizedValue * chartHeight;
      if (height < 0.5) continue; // Skip near-silent bins
      
      const x = padding.left + (i / numBins) * chartWidth;
      const y = padding.top + chartHeight - height;
      
      ctx.fillRect(x, y, barWidth - 0.5, height);
    }
    
    // Highlight Peak Frequency
    if (peakFrequency !== null && peakFrequency <= this.maxDisplayFreq) {
      const x = padding.left + (peakFrequency / this.maxDisplayFreq) * chartWidth;
      
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.strokeStyle = '#e53935';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Label
      ctx.fillStyle = '#e53935';
      ctx.textAlign = 'left';
      ctx.font = 'bold 12px sans-serif';
      
      const labelX = x + 5 > this.width - 100 ? x - 100 : x + 5;
      
      ctx.fillText(`${peakFrequency.toFixed(1)} Hz`, labelX, padding.top + 10);
    }
  }
}

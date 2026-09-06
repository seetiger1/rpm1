export class RPMGauge {
  /**
   * Initialize the RPM Gauge
   * @param {string} canvasId - The ID of the canvas element
   */
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`Canvas with ID ${canvasId} not found`);
    
    this.ctx = this.canvas.getContext('2d');
    
    // Offscreen canvas for static rendering
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    
    this.minRPM = 0;
    this.maxRPM = 4000;
    
    // Angles in radians (0 is 3 o'clock, clockwise)
    // 135 degrees (bottom left) to 405 degrees (bottom right, 45 + 360)
    this.startAngle = Math.PI * 0.75;
    this.endAngle = Math.PI * 2.25;
    this.sweepAngle = this.endAngle - this.startAngle;
    
    this.currentRPM = 0;
    this.targetRPM = 0;
    
    this.isRendering = false;
    this.renderLoop = null;
    
    // Bind methods
    this.render = this.render.bind(this);
    this.resize = this.resize.bind(this);
    
    // Resize observer
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas.parentElement);
    
    // Initial setup
    this.resize();
  }
  
  /**
   * Handle canvas resizing and scale for devicePixelRatio
   */
  resize() {
    const parent = this.canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    
    // Set size maintaining aspect ratio (square)
    const size = Math.min(rect.width, rect.height);
    
    const dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    
    this.offscreenCanvas.width = this.canvas.width;
    this.offscreenCanvas.height = this.canvas.height;
    
    this.ctx.scale(dpr, dpr);
    this.offscreenCtx.scale(dpr, dpr);
    
    this.width = size;
    this.height = size;
    
    this.drawStaticLayer();
  }
  
  /**
   * Draw the static background of the gauge
   */
  drawStaticLayer() {
    const ctx = this.offscreenCtx;
    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const radius = Math.min(centerX, centerY) * 0.9;
    
    ctx.clearRect(0, 0, this.width, this.height);
    
    // Background face
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#d0d0d0';
    ctx.shadowBlur = 10;
    ctx.shadowOffsetY = 2;
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    
    // Border
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#d0d0d0';
    ctx.stroke();
    
    // Color Arcs
    const drawArc = (startRPM, endRPM, color) => {
      const start = this.startAngle + (startRPM / this.maxRPM) * this.sweepAngle;
      const end = this.startAngle + (endRPM / this.maxRPM) * this.sweepAngle;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.85, start, end);
      ctx.lineWidth = radius * 0.08;
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    
    drawArc(0, 2500, '#4caf50'); // Green
    drawArc(2500, 3200, '#ff9800'); // Amber
    drawArc(3200, 4000, '#e53935'); // Red
    
    // Ticks and Labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `bold ${radius * 0.12}px sans-serif`;
    ctx.fillStyle = '#1a1a2e';
    
    for (let rpm = 0; rpm <= this.maxRPM; rpm += 100) {
      const isMajor = rpm % 500 === 0;
      const angle = this.startAngle + (rpm / this.maxRPM) * this.sweepAngle;
      
      const innerRadius = isMajor ? radius * 0.65 : radius * 0.73;
      const outerRadius = radius * 0.78;
      
      const startX = centerX + Math.cos(angle) * innerRadius;
      const startY = centerY + Math.sin(angle) * innerRadius;
      const endX = centerX + Math.cos(angle) * outerRadius;
      const endY = centerY + Math.sin(angle) * outerRadius;
      
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.lineWidth = isMajor ? 3 : 1.5;
      ctx.strokeStyle = '#1a1a2e';
      ctx.stroke();
      
      if (isMajor) {
        const textRadius = radius * 0.52;
        const textX = centerX + Math.cos(angle) * textRadius;
        const textY = centerY + Math.sin(angle) * textRadius;
        
        ctx.save();
        ctx.translate(textX, textY);
        // Ensure tabular numbers if the browser supports it
        ctx.font = `bold ${radius * 0.12}px "SF Pro Display", -apple-system, sans-serif`;
        ctx.fillText(rpm.toString(), 0, 0);
        ctx.restore();
      }
    }
    
    // Center Text
    ctx.font = `bold ${radius * 0.15}px sans-serif`;
    ctx.fillStyle = '#1a1a2e';
    ctx.fillText('RPM', centerX, centerY + radius * 0.4);
    
    // Small center base
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.1, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a2e';
    ctx.fill();
  }
  
  /**
   * Set target RPM
   * @param {number} rpm 
   */
  setRPM(rpm) {
    this.targetRPM = Math.max(this.minRPM, Math.min(rpm, this.maxRPM));
  }
  
  /**
   * Render frame
   */
  render() {
    if (!this.isRendering) return;
    
    // Interpolate
    this.currentRPM += (this.targetRPM - this.currentRPM) * 0.12;
    
    this.ctx.clearRect(0, 0, this.width, this.height);
    
    // Draw static background
    this.ctx.drawImage(this.offscreenCanvas, 0, 0, this.width, this.height);
    
    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const radius = Math.min(centerX, centerY) * 0.9;
    
    // Draw needle
    const currentAngle = this.startAngle + (this.currentRPM / this.maxRPM) * this.sweepAngle;
    
    this.ctx.save();
    this.ctx.translate(centerX, centerY);
    this.ctx.rotate(currentAngle);
    
    this.ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    this.ctx.shadowBlur = 5;
    this.ctx.shadowOffsetY = 2;
    
    this.ctx.beginPath();
    this.ctx.moveTo(-radius * 0.04, 0);
    this.ctx.lineTo(0, -radius * 0.8);
    this.ctx.lineTo(radius * 0.04, 0);
    this.ctx.lineTo(0, radius * 0.1); // Small tail
    this.ctx.fillStyle = '#0066cc';
    this.ctx.fill();
    
    this.ctx.restore();
    
    // Center cap
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius * 0.05, 0, Math.PI * 2);
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.fill();
    
    this.renderLoop = requestAnimationFrame(this.render);
  }
  
  /**
   * Start rendering loop
   */
  start() {
    if (!this.isRendering) {
      this.isRendering = true;
      this.render();
    }
  }
  
  /**
   * Stop rendering loop
   */
  stop() {
    this.isRendering = false;
    if (this.renderLoop) {
      cancelAnimationFrame(this.renderLoop);
      this.renderLoop = null;
    }
  }
}

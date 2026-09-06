# Marine Diesel RPM Meter & Video Analyzer

A web-based and Python-powered toolset for measuring marine diesel engine RPM using **acoustic frequency analysis** and **optical video tracking**. Designed specifically for boat inboard diesel engines (such as Volvo Penta, Yanmar, Bukh, etc.).

---

## Features

### 1. Web Acoustic RPM Meter (`index.html`)
- **Progressive Web App (PWA)**: Works offline and can be installed on iOS & Android devices.
- **Real-Time Acoustic Analysis**: Uses the Web Audio API to capture engine sound via microphone or uploaded audio/video files.
- **Harmonic Product Spectrum (HPS)**: Robust fundamental frequency ($f_0$) extraction using dB-domain HPS with quadratic parabolic peak interpolation.
- **Configurable Engine Presets**:
  - Stroke modes: 4-Stroke (0.5 combustion pulses/rev/cyl) or 2-Stroke.
  - Cylinder selection (1 to 8 cylinders).
  - Target RPM ranges: Idle (*Standgas*, 400–800 RPM), Cruise (*Fahrt*, 800–2000 RPM), Full Load (*Volllast*, 1800–3500 RPM), and Auto.
- **Responsive Dashboard**: Analog gauge, digital readout, confidence indicator, and real-time audio spectrum visualization.

### 2. Web Video RPM Analyzer (`video.html`)
- **Flywheel Optical Marker Tracking**: Calculates engine speed from video recordings of a spinning flywheel marked with a white stripe.
- **Interactive Flywheel Calibration**:
  - Drag-and-drop or select engine video files (`.MOV`, `.MP4`, `.WEBM`).
  - Interactive overlay to position flywheel center and radius.
  - Multi-point Regions of Interest (ROIs) along the flywheel rim.
- **In-Browser Signal Processing**:
  - Extracts per-frame brightness fluctuations from multiple ROIs.
  - Sliding-window FFT with quadratic peak interpolation.
  - Real-time interactive timeline and RPM progression chart.

### 3. Python Video RPM Analyzer CLI (`video_rpm_analyzer.py`)
- Standalone command-line video processing pipeline using OpenCV and NumPy.
- Automatic flywheel circle detection (Hough Circles) or manual ROI specification.
- Live camera stream analysis or batch video file processing.
- Generates RPM summary plots, confidence metrics, and exported CSV data.

---

## Project Structure

```text
├── index.html               # Audio RPM Meter (PWA interface)
├── video.html               # Web-based Video Flywheel RPM Analyzer
├── manifest.json            # PWA manifest for offline mobile installation
├── sw.js                    # Service Worker for offline asset caching
├── css/
│   ├── style.css            # Styles for Audio RPM interface & gauges
│   └── video-rpm.css        # Styles for Video RPM interface & controls
├── js/
│   ├── app.js               # Main Audio App controller & UI logic
│   ├── audio-engine.js      # Web Audio API capture, filtering & FFT
│   ├── dsp.js               # Digital Signal Processing (HPS, peak detection)
│   ├── gauge.js             # Canvas-based analog RPM gauge
│   ├── spectrum.js          # Real-time frequency spectrum renderer
│   └── video-rpm.js         # In-browser video frame processing & FFT pipeline
├── icons/                   # PWA application icons
├── video_rpm_analyzer.py    # Python CLI video analyzer (OpenCV + FFT)
└── test_*.py                # DSP benchmark and unit testing scripts
```

---

## Getting Started

### Running the Web Application Locally

Because modern browsers enforce security policies for Web Audio (Microphone) and Web Workers/Service Workers, serve the files via a local HTTP server rather than opening `index.html` directly via `file://`.

Using Python:
```bash
# Start a local web server in this directory
python3 -m http.server 8000
```
Open your browser and navigate to:
- **Audio RPM Meter**: [http://localhost:8000/index.html](http://localhost:8000/index.html)
- **Video RPM Analyzer**: [http://localhost:8000/video.html](http://localhost:8000/video.html)

---

### Running the Python Video Analyzer CLI

#### Requirements
- Python 3.8+
- OpenCV (`opencv-python` or `opencv-python-headless`)
- NumPy
- Matplotlib (optional, for `--plot`)
- SciPy (optional, for filter benchmarks)

Install dependencies:
```bash
pip install opencv-python numpy matplotlib scipy
```

#### Usage Examples
```bash
# Analyze a video file
python video_rpm_analyzer.py engine_flywheel.mov

# Analyze with RPM plot generation
python video_rpm_analyzer.py engine_flywheel.mov --plot

# Adjust sliding analysis window (in seconds)
python video_rpm_analyzer.py engine_flywheel.mov --window 3.0 --plot

# Analyze live from webcam / connected capture card
python video_rpm_analyzer.py --live 0
```

---

## How It Works

### Acoustic Principle
A 4-stroke engine fires each cylinder once every two crankshaft revolutions. The acoustic firing frequency $f_{fire}$ relates to engine RPM by:
$$\text{RPM} = \frac{f_{fire} \times 60 \times 2}{N_{cylinders}}$$

Using the **Harmonic Product Spectrum (HPS)**, the dominant fundamental frequency is extracted from noisy engine room audio even when the fundamental harmonic is attenuated by background engine rattle, exhaust mufflers, or hull vibration.

### Optical Flywheel Principle
A single white line painted across the flywheel rim creates a sharp brightness peak every time it traverses an observation region (ROI). By extracting brightness series across multiple ROIs and computing a sliding-window FFT, the engine rotation frequency $f_{rot}$ is measured directly:
$$\text{RPM} = f_{rot} \times 60$$

---

## License
MIT License. Open source and free to adapt for marine, automotive, and DIY engine diagnostic applications.

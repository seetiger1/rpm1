#!/usr/bin/env python3
"""
Video-Based RPM Analyzer for Flywheel with White Stripe Marker
==============================================================

Measures engine RPM by video-analyzing a spinning flywheel that has
one white stripe painted on it. The stripe creates periodic brightness
spikes at fixed observation points, which are detected via FFT.

Algorithm:
    1. Auto-detect flywheel center and radius via Hough circle detection
    2. Place multiple ROI sample points around the flywheel's outer face
    3. Extract per-frame brightness at each ROI
    4. Sliding-window FFT to find the dominant rotation frequency
    5. Best-confidence ROI selection per window for robustness
    6. Adaptive center re-tracking to handle camera movement

Usage:
    python video_rpm_analyzer.py <video_file> [options]

Examples:
    python video_rpm_analyzer.py IMG_5428.MOV
    python video_rpm_analyzer.py IMG_5440.MOV --plot
    python video_rpm_analyzer.py IMG_5428.MOV --plot --window 3.0
    python video_rpm_analyzer.py --live 0   # live from camera index 0
"""

import argparse
import sys
import os
import time
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python-headless is required.")
    print("Install with: pip install opencv-python-headless")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Flywheel Detection
# ---------------------------------------------------------------------------

def detect_flywheel_center(frame_gray, min_radius=200, max_radius=500):
    """
    Detect the flywheel circle in a single grayscale frame.

    Returns (cx, cy, radius) or None if not found.
    """
    blurred = cv2.GaussianBlur(frame_gray, (15, 15), 3)
    h, w = frame_gray.shape

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=2, minDist=300,
        param1=120, param2=60,
        minRadius=min_radius, maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Filter: center must be reasonably within the image, radius must be large
    candidates = []
    for c in circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if (0.1 * w < cx < 0.9 * w and
                0.2 * h < cy < 0.85 * h and
                r > min_radius):
            candidates.append((cx, cy, r))

    if not candidates:
        return None

    # Prefer the largest circle that is reasonably centered
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[0]


def detect_flywheel_robust(video_path, n_samples=20, min_radius=200, max_radius=500):
    """
    Robustly detect flywheel center by averaging detections over multiple frames.

    Returns (cx, cy, radius) using median of all successful detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n_samples)

    all_cx, all_cy, all_r = [], [], []

    for i in range(0, min(total, n_samples * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detect_flywheel_center(gray, min_radius, max_radius)

        if result:
            cx, cy, r = result
            all_cx.append(cx)
            all_cy.append(cy)
            all_r.append(r)

    cap.release()

    if len(all_cx) >= 3:
        return int(np.median(all_cx)), int(np.median(all_cy)), int(np.median(all_r))
    elif len(all_cx) > 0:
        return int(np.mean(all_cx)), int(np.mean(all_cy)), int(np.mean(all_r))
    else:
        return None


# ---------------------------------------------------------------------------
# Brightness Extraction
# ---------------------------------------------------------------------------

def get_roi_positions(cx, cy, radius, n_rois=8, ring_fraction=0.78):
    """
    Compute ROI sample positions evenly spaced around the flywheel face.

    Args:
        cx, cy: Flywheel center coordinates.
        radius: Flywheel radius in pixels.
        n_rois: Number of sample points around the circle.
        ring_fraction: Fraction of radius for the sampling ring (0.78 = outer face).

    Returns:
        List of (x, y) positions.
    """
    sample_r = int(radius * ring_fraction)
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)
    positions = [
        (int(cx + sample_r * np.cos(a)), int(cy + sample_r * np.sin(a)))
        for a in angles
    ]
    return positions


def extract_roi_brightness(gray_frame, positions, patch_half_size=12):
    """
    Extract mean brightness at each ROI position from a grayscale frame.

    Returns a list of float brightness values.
    """
    h, w = gray_frame.shape
    ps = patch_half_size
    values = []

    for sx, sy in positions:
        y1 = max(0, sy - ps)
        y2 = min(h, sy + ps)
        x1 = max(0, sx - ps)
        x2 = min(w, sx + ps)

        if y2 > y1 and x2 > x1:
            roi = gray_frame[y1:y2, x1:x2]
            values.append(float(roi.mean()))
        else:
            values.append(0.0)

    return values


# ---------------------------------------------------------------------------
# RPM Computation
# ---------------------------------------------------------------------------

def windowed_fft_rpm(signal, fps, window_sec=2.0, step_sec=0.25,
                     min_rpm=200, max_rpm=3000):
    """
    Compute RPM over time using sliding-window FFT.

    Args:
        signal: 1D array of brightness values.
        fps: Video framerate.
        window_sec: FFT window size in seconds.
        step_sec: Step size between windows in seconds.
        min_rpm: Minimum RPM to consider.
        max_rpm: Maximum RPM to consider (capped by Nyquist).

    Returns:
        times: Array of center-timestamps for each window.
        rpms: Array of detected RPM values.
        confidences: Array of SNR-like confidence values.
    """
    window_frames = int(window_sec * fps)
    step_frames = max(1, int(step_sec * fps))

    # Cap max_rpm by Nyquist limit
    nyquist_rpm = (fps / 2.0) * 60.0
    max_rpm = min(max_rpm, nyquist_rpm * 0.95)

    min_freq = min_rpm / 60.0
    max_freq = max_rpm / 60.0

    rpms = []
    times = []
    confidences = []

    i = 0
    while i + window_frames <= len(signal):
        chunk = signal[i:i + window_frames].copy()

        # Remove DC (mean) and linear trend
        chunk -= np.mean(chunk)

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(len(chunk))
        windowed = chunk * window

        # FFT
        fft_vals = np.fft.rfft(windowed)
        fft_freqs = np.fft.rfftfreq(len(windowed), d=1.0 / fps)
        magnitude = np.abs(fft_vals)

        # Search within RPM range
        mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)

        if mask.any():
            masked_mag = magnitude.copy()
            masked_mag[~mask] = 0

            peak_idx = np.argmax(masked_mag)
            peak_freq = fft_freqs[peak_idx]
            peak_mag = masked_mag[peak_idx]

            # Confidence: peak-to-mean ratio (SNR-like)
            mean_mag = np.mean(magnitude[mask])
            confidence = peak_mag / (mean_mag + 1e-10)

            rpms.append(peak_freq * 60.0)
            confidences.append(confidence)
        else:
            rpms.append(0.0)
            confidences.append(0.0)

        times.append((i + window_frames / 2.0) / fps)
        i += step_frames

    return np.array(times), np.array(rpms), np.array(confidences)


def compute_overall_rpm(all_rpm_traces, all_conf_traces):
    """
    Combine RPM estimates from multiple ROIs using confidence-weighted selection.

    For each time window, selects the RPM from the ROI with highest confidence.
    Returns the per-window best RPMs and the overall median.
    """
    all_rpm_traces = np.array(all_rpm_traces)
    all_conf_traces = np.array(all_conf_traces)

    n_rois, n_windows = all_rpm_traces.shape

    # Per-window: pick the ROI with highest confidence
    best_rpms = []
    for t in range(n_windows):
        best_roi = np.argmax(all_conf_traces[:, t])
        best_rpms.append(all_rpm_traces[best_roi, t])

    best_rpms = np.array(best_rpms)

    # Remove outliers using IQR
    if len(best_rpms) > 5:
        q1, q3 = np.percentile(best_rpms, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            valid = (best_rpms >= q1 - 1.5 * iqr) & (best_rpms <= q3 + 1.5 * iqr)
            filtered = best_rpms[valid]
            if len(filtered) > 0:
                overall = np.median(filtered)
            else:
                overall = np.median(best_rpms)
        else:
            overall = np.median(best_rpms)
    else:
        overall = np.median(best_rpms)

    return best_rpms, overall


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_video(video_path, window_sec=2.0, step_sec=0.25,
                  min_rpm=200, max_rpm=3000,
                  n_rois=8, ring_fraction=0.78, patch_size=12,
                  center_retrack_interval=0,
                  manual_center=None,
                  verbose=True):
    """
    Full RPM analysis pipeline for a video file.

    Args:
        video_path: Path to the video file.
        window_sec: FFT window size in seconds.
        step_sec: Step between FFT windows in seconds.
        min_rpm: Minimum expected RPM.
        max_rpm: Maximum expected RPM.
        n_rois: Number of ROI sample points around flywheel.
        ring_fraction: Fraction of flywheel radius for ROI ring.
        patch_size: Half-size of each ROI patch in pixels.
        center_retrack_interval: Re-detect flywheel center every N frames
                                 (0 = detect once). Handles camera movement.
        manual_center: Tuple (cx, cy, radius) to override auto-detection.
        verbose: Print progress info.

    Returns:
        Dictionary with all results.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if verbose:
        print(f"Video: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f}s ({total_frames} frames)")
        nyquist_rpm = (fps / 2.0) * 60.0
        print(f"  Nyquist limit: {nyquist_rpm:.0f} RPM (at {fps:.0f} fps)")
        print()

    # --- Step 1: Detect flywheel ---
    if manual_center:
        cx, cy, radius = manual_center
        if verbose:
            print(f"  Using manual flywheel: center=({cx}, {cy}), radius={radius}")
    else:
        if verbose:
            print("  Detecting flywheel position...")
        result = detect_flywheel_robust(video_path)
        if result is None:
            print("  ERROR: Could not detect flywheel. Use --center CX CY R")
            return None
        cx, cy, radius = result
        if verbose:
            print(f"  Flywheel detected: center=({cx}, {cy}), radius={radius}")

    # --- Step 2: Extract brightness signals ---
    if verbose:
        print(f"  Extracting brightness from {n_rois} ROI points...")

    cap = cv2.VideoCapture(video_path)
    positions = get_roi_positions(cx, cy, radius, n_rois, ring_fraction)
    all_signals = [[] for _ in range(n_rois)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive center re-tracking for camera movement
        if (center_retrack_interval > 0 and
                frame_idx > 0 and
                frame_idx % center_retrack_interval == 0):
            new_center = detect_flywheel_center(gray)
            if new_center:
                new_cx, new_cy, new_r = new_center
                # Smooth update (weighted average to avoid jitter)
                cx = int(0.7 * cx + 0.3 * new_cx)
                cy = int(0.7 * cy + 0.3 * new_cy)
                radius = int(0.7 * radius + 0.3 * new_r)
                positions = get_roi_positions(cx, cy, radius, n_rois, ring_fraction)

        # Extract brightness at each ROI
        values = extract_roi_brightness(gray, positions, patch_size)
        for j, v in enumerate(values):
            all_signals[j].append(v)

        frame_idx += 1

    cap.release()

    if verbose:
        print(f"  Processed {frame_idx} frames")

    # Convert to numpy arrays
    signals = [np.array(s) for s in all_signals]

    # --- Step 3: Windowed FFT on each ROI ---
    if verbose:
        print(f"  Computing windowed FFT (window={window_sec}s, step={step_sec}s)...")

    all_rpm_traces = []
    all_conf_traces = []

    for sig in signals:
        times, rpms, confs = windowed_fft_rpm(
            sig, fps, window_sec, step_sec, min_rpm, max_rpm
        )
        all_rpm_traces.append(rpms)
        all_conf_traces.append(confs)

    if len(times) == 0:
        print("  ERROR: Video too short for FFT window. Reduce --window size.")
        return None

    # --- Step 4: Combine ROI results ---
    best_rpms, overall_rpm = compute_overall_rpm(all_rpm_traces, all_conf_traces)

    if verbose:
        print()
        print(f"  ╔══════════════════════════════════════╗")
        print(f"  ║  RESULT: {overall_rpm:7.0f} RPM (median)       ║")
        print(f"  ╚══════════════════════════════════════╝")
        print()

        # Show RPM over time
        print("  RPM over time:")
        for i, (t, rpm) in enumerate(zip(times, best_rpms)):
            bar_len = int(rpm / 50)
            bar = "█" * min(bar_len, 60)
            print(f"    {t:6.1f}s: {rpm:7.0f} RPM  {bar}")

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": frame_idx,
        "duration": duration,
        "flywheel_center": (cx, cy),
        "flywheel_radius": radius,
        "times": times,
        "best_rpms": best_rpms,
        "overall_rpm": overall_rpm,
        "all_rpm_traces": np.array(all_rpm_traces),
        "all_conf_traces": np.array(all_conf_traces),
        "signals": signals,
        "roi_positions": positions,
        "nyquist_rpm": (fps / 2.0) * 60.0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(result, output_path=None):
    """Generate diagnostic plots for the RPM analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available. Skipping plots.")
        return

    video_name = os.path.basename(result["video_path"])
    fps = result["fps"]
    times = result["times"]
    best_rpms = result["best_rpms"]
    overall_rpm = result["overall_rpm"]
    signals = result["signals"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Plot 1: Raw brightness signals ---
    ax = axes[0]
    time_axis = np.arange(len(signals[0])) / fps
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))
    for i, sig in enumerate(signals):
        ax.plot(time_axis, sig, linewidth=0.4, alpha=0.6, color=colors[i],
                label=f"ROI {i}")
    ax.set_title(f"{video_name} — Raw Brightness Signals ({len(signals)} ROIs)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Brightness (grayscale)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: RPM over time ---
    ax = axes[1]
    ax.plot(times, best_rpms, "b.-", linewidth=2, markersize=5,
            label="Best-confidence RPM")
    ax.axhline(overall_rpm, color="green", linestyle="--", linewidth=2,
               label=f"Median: {overall_rpm:.0f} RPM")
    ax.fill_between(times, overall_rpm * 0.9, overall_rpm * 1.1,
                     alpha=0.15, color="green", label="±10% band")
    ax.set_title(f"RPM Over Time — Overall: {overall_rpm:.0f} RPM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.set_ylim(0, max(best_rpms.max() * 1.3, overall_rpm * 1.5))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Plot 3: FFT spectrum of best ROI ---
    ax = axes[2]
    # Find the ROI with highest average confidence
    avg_confs = result["all_conf_traces"].mean(axis=1)
    best_roi_idx = np.argmax(avg_confs)
    best_signal = signals[best_roi_idx]

    # Full-video FFT
    sig = best_signal - np.mean(best_signal)
    window = np.hanning(len(sig))
    fft_vals = np.fft.rfft(sig * window)
    fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    magnitude = np.abs(fft_vals)

    # Plot in RPM units
    rpm_axis = fft_freqs * 60.0
    mask = (rpm_axis >= 100) & (rpm_axis <= result["nyquist_rpm"])
    ax.plot(rpm_axis[mask], magnitude[mask], "r-", linewidth=0.8)
    ax.axvline(overall_rpm, color="green", linestyle="--", linewidth=2,
               label=f"Detected: {overall_rpm:.0f} RPM")
    ax.set_title(f"FFT Spectrum (best ROI #{best_roi_idx})")
    ax.set_xlabel("RPM")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        base = os.path.splitext(video_name)[0]
        output_path = f"{base}_rpm_analysis.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Annotated Frame Export
# ---------------------------------------------------------------------------

def export_annotated_frame(video_path, result, output_path=None):
    """Export a single frame with flywheel detection and ROI positions annotated."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Seek to 1 second in
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return

    cx, cy = result["flywheel_center"]
    r = result["flywheel_radius"]
    positions = result["roi_positions"]
    rpm = result["overall_rpm"]

    # Draw flywheel circle
    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Draw ROI positions
    for i, (px, py) in enumerate(positions):
        cv2.rectangle(frame, (px - 12, py - 12), (px + 12, py + 12),
                       (0, 255, 255), 2)
        cv2.putText(frame, str(i), (px + 15, py + 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw RPM text
    cv2.putText(frame, f"RPM: {rpm:.0f}", (20, 60),
                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_annotated.jpg"

    cv2.imwrite(output_path, frame)
    print(f"  Annotated frame saved: {output_path}")


# ---------------------------------------------------------------------------
# Live Camera Analysis (Skeleton)
# ---------------------------------------------------------------------------

def analyze_live(camera_index=0, min_rpm=200, max_rpm=3000,
                 n_rois=8, ring_fraction=0.78, window_sec=2.0):
    """
    Live RPM analysis from a camera feed.

    This is a working skeleton — it continuously reads frames from the camera,
    detects the flywheel, and computes RPM in real-time using a rolling buffer.

    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default assumption
    print(f"Camera opened: FPS={fps:.1f}")
    print(f"Nyquist RPM limit: {(fps/2)*60:.0f}")
    print("Press 'q' to quit (if display is available)\n")

    # Rolling buffer for brightness signals
    buffer_size = int(window_sec * fps)
    buffers = [[] for _ in range(n_rois)]
    positions = None
    cx, cy, radius = 0, 0, 0
    frame_count = 0
    detection_interval = int(fps * 2)  # Re-detect every 2 seconds

    current_rpm = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect/update flywheel position periodically
        if frame_count % detection_interval == 0 or positions is None:
            result = detect_flywheel_center(gray)
            if result:
                if positions is not None:
                    # Smooth update
                    cx = int(0.7 * cx + 0.3 * result[0])
                    cy = int(0.7 * cy + 0.3 * result[1])
                    radius = int(0.7 * radius + 0.3 * result[2])
                else:
                    cx, cy, radius = result
                positions = get_roi_positions(cx, cy, radius, n_rois, ring_fraction)
                print(f"  Flywheel: center=({cx},{cy}), radius={radius}")

        if positions is None:
            frame_count += 1
            continue

        # Extract brightness
        values = extract_roi_brightness(gray, positions)
        for j, v in enumerate(values):
            buffers[j].append(v)
            # Keep only last buffer_size values
            if len(buffers[j]) > buffer_size:
                buffers[j] = buffers[j][-buffer_size:]

        # Compute RPM when buffer is full
        if len(buffers[0]) >= buffer_size:
            best_rpm_val = 0
            best_conf = 0
            for buf in buffers:
                sig = np.array(buf)
                _, rpms, confs = windowed_fft_rpm(
                    sig, fps, window_sec=window_sec, step_sec=window_sec,
                    min_rpm=min_rpm, max_rpm=max_rpm
                )
                if len(rpms) > 0 and confs[0] > best_conf:
                    best_conf = confs[0]
                    best_rpm_val = rpms[0]

            current_rpm = best_rpm_val

            # Print RPM periodically
            if frame_count % int(fps) == 0:
                print(f"  [{frame_count/fps:6.1f}s] RPM: {current_rpm:7.0f}")

        # Try to display frame with overlay (if display is available)
        try:
            display_frame = frame.copy()
            if positions:
                cv2.circle(display_frame, (cx, cy), radius, (0, 255, 0), 2)
                for px, py in positions:
                    cv2.circle(display_frame, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(display_frame, f"RPM: {current_rpm:.0f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

            # Resize for display if too large
            dh, dw = display_frame.shape[:2]
            if dh > 800:
                scale = 800.0 / dh
                display_frame = cv2.resize(display_frame,
                                           (int(dw * scale), int(dh * scale)))

            cv2.imshow("RPM Analyzer", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # No display available (headless mode)
            pass

        frame_count += 1

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"\nFinal RPM: {current_rpm:.0f}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Video-based RPM Analyzer for flywheel with white stripe marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s RPM-Testdateien/IMG_5428.MOV
  %(prog)s RPM-Testdateien/IMG_5440.MOV --plot
  %(prog)s RPM-Testdateien/IMG_5428.MOV --window 3.0 --min-rpm 300 --max-rpm 2000
  %(prog)s --live 0
        """,
    )

    # Input source
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--live", type=int, metavar="CAM_IDX",
                        help="Live analysis from camera index (e.g., 0)")

    # Analysis parameters
    parser.add_argument("--window", type=float, default=2.0,
                        help="FFT window size in seconds (default: 2.0)")
    parser.add_argument("--step", type=float, default=0.25,
                        help="FFT step size in seconds (default: 0.25)")
    parser.add_argument("--min-rpm", type=int, default=200,
                        help="Minimum expected RPM (default: 200)")
    parser.add_argument("--max-rpm", type=int, default=3000,
                        help="Maximum expected RPM (default: 3000)")
    parser.add_argument("--n-rois", type=int, default=8,
                        help="Number of ROI sample points (default: 8)")
    parser.add_argument("--ring-frac", type=float, default=0.78,
                        help="ROI ring as fraction of flywheel radius (default: 0.78)")

    # Manual flywheel override
    parser.add_argument("--center", type=int, nargs=3, metavar=("CX", "CY", "R"),
                        help="Manual flywheel center and radius (CX CY RADIUS)")

    # Adaptive tracking
    parser.add_argument("--retrack", type=int, default=0,
                        help="Re-detect flywheel center every N frames (0=off, default: 0)")

    # Output
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--annotate", action="store_true",
                        help="Export annotated frame showing detected flywheel and ROIs")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (default: current dir)")

    args = parser.parse_args()

    # Validate arguments
    if args.live is not None:
        analyze_live(
            camera_index=args.live,
            min_rpm=args.min_rpm,
            max_rpm=args.max_rpm,
            n_rois=args.n_rois,
            ring_fraction=args.ring_frac,
            window_sec=args.window,
        )
        return

    if not args.video:
        parser.error("Either a video file or --live CAM_IDX is required")

    if not os.path.isfile(args.video):
        parser.error(f"Video file not found: {args.video}")

    # Run analysis
    manual_center = tuple(args.center) if args.center else None

    result = analyze_video(
        args.video,
        window_sec=args.window,
        step_sec=args.step,
        min_rpm=args.min_rpm,
        max_rpm=args.max_rpm,
        n_rois=args.n_rois,
        ring_fraction=args.ring_frac,
        patch_size=12,
        center_retrack_interval=args.retrack,
        manual_center=manual_center,
        verbose=True,
    )

    if result is None:
        sys.exit(1)

    # Generate outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]

    if args.plot:
        plot_path = os.path.join(args.output_dir, f"{base}_rpm_analysis.png")
        generate_plots(result, output_path=plot_path)

    if args.annotate:
        annot_path = os.path.join(args.output_dir, f"{base}_annotated.jpg")
        export_annotated_frame(args.video, result, output_path=annot_path)


if __name__ == "__main__":
    main()

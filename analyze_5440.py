import numpy as np
import librosa
import sys
import scipy.signal

filename = 'RPM-Testdateien/IMG_5440.MOV'
try:
    y, sr = librosa.load(filename, sr=44100, duration=10.0) # load first 10 secs
except Exception as e:
    print(f"Error loading {filename}: {e}")
    sys.exit(1)

# Calculate power spectrogram
D = np.abs(librosa.stft(y, n_fft=8192))
mean_mag = np.mean(D, axis=1)

freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)

# Find top peaks
peaks, _ = scipy.signal.find_peaks(mean_mag, distance=3)
peak_mags = mean_mag[peaks]
top_peaks_idx = peaks[np.argsort(peak_mags)[::-1]][:20]

print("Top 20 raw frequency peaks in IMG_5440.MOV (sorted by magnitude):")
for idx in sorted(top_peaks_idx):
    f = freqs[idx]
    rpm_4cyl = f * 120 / 4
    print(f"{f:.2f} Hz  ->  {rpm_4cyl:.0f} RPM (Mag: {mean_mag[idx]:.2f})")

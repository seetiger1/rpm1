import numpy as np
import librosa
import scipy.signal

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100, duration=10.0)
D = np.abs(librosa.stft(y, n_fft=8192))
mean_mag = np.mean(D, axis=1)
freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
peaks, _ = scipy.signal.find_peaks(mean_mag, distance=3)
peak_mags = mean_mag[peaks]
top_peaks_idx = peaks[np.argsort(peak_mags)[::-1]][:10]

print("Top 10 raw frequency peaks in 1-5428.mp3:")
for idx in sorted(top_peaks_idx):
    f = freqs[idx]
    print(f"{f:.2f} Hz (Mag: {mean_mag[idx]:.2f})")

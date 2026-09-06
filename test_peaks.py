import numpy as np
import librosa
from scipy.signal import find_peaks

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
start_sample = len(y) // 2
frame = y[start_sample:start_sample + 16384]

stft = librosa.stft(frame, n_fft=8192, center=False)
mag = np.mean(np.abs(stft), axis=1) # average over a few frames
freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)

db = 20 * np.log10(mag + 1e-10)
peaks, properties = find_peaks(db, height=-80, distance=3)

print("Top peaks in dB (under 300Hz):")
peak_data = []
for p in peaks:
    if freqs[p] < 300:
        peak_data.append((freqs[p], db[p]))

# sort by frequency to see harmonic series
peak_data.sort(key=lambda x: x[0])
for f, m in peak_data:
    print(f"Freq: {f:.1f} Hz, Mag: {m:.1f} dB")

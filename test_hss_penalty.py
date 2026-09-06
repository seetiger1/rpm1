import numpy as np
import librosa
from scipy.signal import find_peaks

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
start_sample = len(y) // 2
frame = y[start_sample:start_sample + 16384]

stft = librosa.stft(frame, n_fft=8192, center=False)
mag = np.mean(np.abs(stft), axis=1)
freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)

# Convert to linear
linear_mag = mag

hss = np.zeros(len(linear_mag))
num_harmonics = 6

for i in range(len(linear_mag)):
    f = freqs[i]
    if f < 10 or f > 100:
        continue
    
    # Harmonic sum
    score = 0
    for h in range(1, num_harmonics + 1):
        idx = i * h
        if idx < len(linear_mag):
            score += linear_mag[idx]
            
    # Apply a penalty for higher frequencies to favor true fundamentals
    # A true fundamental at 16 Hz will have a strong HSS.
    # Its 2nd harmonic at 32 Hz will also have a strong HSS (using 32, 64, 96...).
    hss[i] = score / np.sqrt(f) # Penalty

peaks, _ = find_peaks(hss)
best_peaks = sorted(peaks, key=lambda p: hss[p], reverse=True)

for p in best_peaks[:5]:
    f = freqs[p]
    rpm = f * 120 / 4
    print(f"Freq: {f:.2f} Hz, RPM: {rpm:.0f}, Score: {hss[p]:.2f}")


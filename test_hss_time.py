import numpy as np
import librosa
from scipy.signal import find_peaks

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
hop = 4096
freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)

rpms = []
for i in range(len(y)//2, len(y) - hop, hop*4):
    frame = y[i:i+8192]
    if len(frame) < 8192:
        break
    stft = librosa.stft(frame, n_fft=8192, center=False)
    linear_mag = np.mean(np.abs(stft), axis=1)
    
    hss = np.zeros(len(linear_mag))
    for j in range(len(linear_mag)):
        f = freqs[j]
        if f < 10 or f > 100:
            continue
        score = 0
        for h in range(1, 7):
            idx = j * h
            if idx < len(linear_mag):
                score += linear_mag[idx]
        hss[j] = score / np.sqrt(f)
        
    peaks, _ = find_peaks(hss)
    if len(peaks) > 0:
        best_p = max(peaks, key=lambda p: hss[p])
        f = freqs[best_p]
        rpm = f * 120 / 4
        rpms.append(rpm)
        print(f"Time {i/sr:.1f}s: RPM = {rpm:.0f}")

print(f"Median RPM: {np.median(rpms):.0f}")

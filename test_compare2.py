import numpy as np
import librosa
import math

y, sr = librosa.load('RPM-Testdateien/IMG_5440.MOV', sr=44100)
hop = 4096
start_sample = len(y)//2
frame = y[start_sample:start_sample + 8192]
window = np.blackman(8192)
stft = np.fft.rfft(frame * window)
mag = np.abs(stft)
spectrum = 20 * np.log10(mag + 1e-10)
spectrum = np.maximum(spectrum, -100)
length = len(spectrum)
hps = np.zeros(length)
for i in range(length): hps[i] = math.pow(10, spectrum[i] / 20)

for i in [4, 6]:
    print(f"\nBin {i} ({i * (44100/2/4096):.2f} Hz):")
    total = 0
    for h in range(1, 7):
        idx = i * h
        val = math.pow(10, spectrum[idx] / 20)
        total += val
        print(f"  h={h} ({idx * (44100/2/4096):.2f} Hz): {val:.2f} (dB: {spectrum[idx]:.2f})")
    print(f"  Total sum: {total:.2f}")

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
for h in range(2, 7):
    for i in range(length):
        idx = i * h
        if idx < length: hps[i] += math.pow(10, spectrum[idx] / 20)
        
# PENALTY 1.0
for i in range(1, length): hps[i] = hps[i] / i

for i in range(length): hps[i] = 20 * math.log10(hps[i] + 1e-10)

freqPerBin = (44100 / 2) / 4096
minBin = max(1, math.ceil(15 / freqPerBin)) 
maxBin = min(4095, math.ceil(500 / freqPerBin))

peaks = []
for i in range(minBin, maxBin + 1):
    peaks.append((i, i * freqPerBin, i * freqPerBin * 120 / 4, hps[i]))

peaks.sort(key=lambda x: x[3], reverse=True)
print("Top 10 HPS bins:")
for p in peaks[:10]:
    print(f"Bin {p[0]}: {p[1]:.2f} Hz -> {p[2]:.0f} RPM (HPS: {p[3]:.2f} dB)")

import numpy as np
import librosa
import math

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
# we take 8192 samples
start_sample = len(y) // 2
frame = y[start_sample:start_sample + 8192]

# mimic JS AnalyserNode
# AnalyserNode applies Blackman window
window = np.blackman(8192)
stft = np.fft.rfft(frame * window)
mag = np.abs(stft)

# Web Audio API does 20 * log10(mag / 8192) roughly, but it scales
spectrum = 20 * np.log10(mag + 1e-10)

length = len(spectrum)
hps = np.zeros(length)
for i in range(length):
    hps[i] = math.pow(10, spectrum[i] / 20)

for h in range(2, 7):
    for i in range(length):
        idx = i * h
        if idx < length:
            hps[i] += math.pow(10, spectrum[idx] / 20)

for i in range(1, length):
    hps[i] = hps[i] / math.sqrt(i)

for i in range(length):
    hps[i] = 20 * math.log10(hps[i] + 1e-10)

# findPeakFrequency JS logic
binCount = 4096
freqPerBin = (44100 / 2) / binCount
minFreq = 15
maxFreq = 500
minBin = max(1, math.floor(minFreq / freqPerBin))
maxBin = min(binCount - 1, math.ceil(maxFreq / freqPerBin))

maxVal = -float('inf')
maxIndex = -1
sum_val = 0
count = 0

for i in range(minBin, maxBin + 1):
    val = hps[i]
    sum_val += val
    count += 1
    if val > maxVal:
        maxVal = val
        maxIndex = i

print(f"JS Logic Peak: Bin {maxIndex} -> {maxIndex * freqPerBin:.2f} Hz -> {maxIndex * freqPerBin * 120 / 4:.0f} RPM")
print(f"Bin 3 raw: {hps[3]:.2f}")
print(f"Bin 6 raw: {hps[6]:.2f}")
print(f"Bin 12 raw: {hps[12]:.2f}")
print(f"Linear Bin 3: {hps[3] * math.sqrt(3):.2f}")
print(f"Linear Bin 6: {hps[6] * math.sqrt(6):.2f}")
print(f"Spec 3: {math.pow(10, spectrum[3]/20):.2f}, Spec 6: {math.pow(10, spectrum[6]/20):.2f}, Spec 9: {math.pow(10, spectrum[9]/20):.2f}, Spec 12: {math.pow(10, spectrum[12]/20):.2f}, Spec 15: {math.pow(10, spectrum[15]/20):.2f}, Spec 18: {math.pow(10, spectrum[18]/20):.2f}")
print(f"Spec 24: {math.pow(10, spectrum[24]/20):.2f}, Spec 30: {math.pow(10, spectrum[30]/20):.2f}, Spec 36: {math.pow(10, spectrum[36]/20):.2f}")

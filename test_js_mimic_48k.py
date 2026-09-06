import numpy as np
import librosa
import math

# resample to 48000
y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=48000)
hop = int(4096 * (48000/44100))

rpms = []
smoothed_mag = None
for start_sample in range(len(y)//2, len(y) - 8192, hop):
    frame = y[start_sample:start_sample + 8192]
    if len(frame) < 8192:
        break
    window = np.blackman(8192)
    stft = np.fft.rfft(frame * window)
    mag = np.abs(stft)
    
    if smoothed_mag is None:
        smoothed_mag = mag
    else:
        smoothed_mag = 0.8 * smoothed_mag + 0.2 * mag
        
    spectrum = 20 * np.log10(smoothed_mag + 1e-10)
    
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
        hps[i] = hps[i] / i
    
    for i in range(length):
        hps[i] = 20 * math.log10(hps[i] + 1e-10)
    
    freqPerBin = (48000 / 2) / 4096
    minBin = max(1, math.ceil(15 / freqPerBin)) 
    maxBin = min(4095, math.ceil(500 / freqPerBin))
    
    maxVal = -float('inf')
    maxIndex = -1
    for i in range(minBin, maxBin + 1):
        if hps[i] > maxVal:
            maxVal = hps[i]
            maxIndex = i
            
    # Parabolic interpolation
    peakFreq = maxIndex * freqPerBin
    if maxIndex > minBin and maxIndex < maxBin:
        alpha = hps[maxIndex - 1]
        beta = hps[maxIndex]
        gamma = hps[maxIndex + 1]
        denom = alpha - 2 * beta + gamma
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
            peakFreq = (maxIndex + p) * freqPerBin

    rpms.append(peakFreq * 120 / 4)

print("Values:")
print([int(r) for r in rpms])

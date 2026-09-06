import numpy as np
import librosa
import math

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
hop = 4096

rpms = []
smoothed_mag = None
for start_sample in range(len(y)//2, len(y) - 8192, hop):
    frame = y[start_sample:start_sample + 8192]
    window = np.blackman(8192)
    stft = np.fft.rfft(frame * window)
    mag = np.abs(stft)
    if smoothed_mag is None: smoothed_mag = mag
    else: smoothed_mag = 0.8 * smoothed_mag + 0.2 * mag
    spectrum = 20 * np.log10(smoothed_mag + 1e-10)
    
    length = len(spectrum)
    hps = np.zeros(length)
    for i in range(length): hps[i] = math.pow(10, spectrum[i] / 20)
    for h in range(2, 4): # ONLY 3 HARMONICS! (h=2, h=3)
        for i in range(length):
            idx = i * h
            if idx < length: hps[i] += math.pow(10, spectrum[idx] / 20)
            
    # PENALTY 1.0
    for i in range(1, length): hps[i] = hps[i] / i
    
    for i in range(length): hps[i] = 20 * math.log10(hps[i] + 1e-10)
    
    freqPerBin = (44100 / 2) / 4096
    minBin = max(1, math.ceil(15 / freqPerBin)) 
    maxBin = min(4095, math.ceil(500 / freqPerBin))
    
    maxVal = -float('inf')
    maxIndex = -1
    for i in range(minBin, maxBin + 1):
        if hps[i] > maxVal:
            maxVal = hps[i]
            maxIndex = i
            
    rpms.append(maxIndex * freqPerBin * 120 / 4)

print("Values:")
print([int(r) for r in rpms])

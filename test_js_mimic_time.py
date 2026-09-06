import numpy as np
import librosa
import math

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
hop = 4096

rpms = []
for start_sample in range(len(y)//2, len(y) - 8192, hop*4):
    frame = y[start_sample:start_sample + 8192]
    
    window = np.blackman(8192)
    stft = np.fft.rfft(frame * window)
    mag = np.abs(stft)
    spectrum = 20 * np.log10(mag + 1e-10)
    
    length = len(spectrum)
    hps = np.zeros(length)
    for i in range(length):
        hps[i] = math.pow(10, spectrum[i] / 20)
    
    for h in range(2, 9): # numHarmonics = 8
        for i in range(length):
            idx = i * h
            if idx < length:
                hps[i] += math.pow(10, spectrum[idx] / 20)
    
    for i in range(1, length):
        hps[i] = hps[i] / math.pow(i, 0.5)
    
    for i in range(length):
        hps[i] = 20 * math.log10(hps[i] + 1e-10)
    
    binCount = 4096
    freqPerBin = (44100 / 2) / binCount
    minFreq = 15
    maxFreq = 500
    minBin = max(1, math.floor(minFreq / freqPerBin))
    maxBin = min(binCount - 1, math.ceil(maxFreq / freqPerBin))
    
    maxVal = -float('inf')
    maxIndex = -1
    for i in range(minBin, maxBin + 1):
        val = hps[i]
        if val > maxVal:
            maxVal = val
            maxIndex = i
            
    rpm = maxIndex * freqPerBin * 120 / 4
    rpms.append(rpm)
    print(f"Time {start_sample/sr:.1f}s: RPM = {rpm:.0f}")

print(f"Median RPM: {np.median(rpms):.0f}")

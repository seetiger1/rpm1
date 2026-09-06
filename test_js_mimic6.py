import numpy as np
import librosa
import math
from scipy.signal import find_peaks

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
hop = 4096

rpms = []
smoothed_mag = None
for start_sample in range(len(y)//2, len(y) - 8192, hop):
    frame = y[start_sample:start_sample + 8192]
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
    
    for h in range(2, 7): # numHarmonics = 6
        for i in range(length):
            idx = i * h
            if idx < length:
                hps[i] += math.pow(10, spectrum[idx] / 20)
    
    # NO PENALTY! We'll use the heuristic instead.
    for i in range(length):
        hps[i] = 20 * math.log10(hps[i] + 1e-10)
    
    freqPerBin = (44100 / 2) / 4096
    minBin = max(1, math.floor(15 / freqPerBin))
    maxBin = min(4095, math.ceil(500 / freqPerBin))
    
    search_array = hps[minBin:maxBin+1]
    maxVal = np.max(search_array)
    
    # find local peaks
    peaks, _ = find_peaks(search_array)
    
    best_bin = -1
    for p in peaks:
        if search_array[p] > maxVal - 6: # within 6 dB of absolute max
            best_bin = p + minBin
            break # pick the first (lowest) one!
            
    if best_bin == -1:
        # fallback
        best_bin = minBin + np.argmax(search_array)
            
    rpms.append(best_bin * freqPerBin * 120 / 4)

print("Values:")
print([int(r) for r in rpms])
print(f"Heuristic RPM: Median {np.median(rpms):.0f} RPM (Mean: {np.mean(rpms):.0f}, Std: {np.std(rpms):.0f})")

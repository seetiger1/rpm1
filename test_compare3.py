import numpy as np
import librosa
import math

def get_peaks(filename, is_linear):
    y, sr = librosa.load(filename, sr=44100)
    hop = 4096
    start_sample = len(y)//2
    frame = y[start_sample:start_sample + 8192]
    window = np.blackman(8192)
    stft = np.fft.rfft(frame * window)
    mag = np.abs(stft)
    spectrum = 20 * np.log10(mag + 1e-10)
    # Clamp to prevent missing fundamental from ruining the sum of dB
    spectrum = np.maximum(spectrum, 0)
    
    length = len(spectrum)
    hps = np.zeros(length)
    
    if is_linear:
        for i in range(length): hps[i] = math.pow(10, spectrum[i] / 20)
        for h in range(2, 7):
            for i in range(length):
                idx = i * h
                if idx < length: hps[i] += math.pow(10, spectrum[idx] / 20)
        # PENALTY
        for i in range(1, length): hps[i] = hps[i] / i
    else:
        for i in range(length): hps[i] = spectrum[i]
        for h in range(2, 7):
            for i in range(length):
                idx = i * h
                if idx < length: hps[i] += spectrum[idx]
        # PENALTY for dB (subtract instead of divide, or just no penalty?)
        for i in range(1, length): hps[i] = hps[i] - 10 * math.log10(i)
        
    freqPerBin = (44100 / 2) / 4096
    minBin = max(1, math.ceil(15 / freqPerBin)) 
    maxBin = min(4095, math.ceil(500 / freqPerBin))
    
    peaks = []
    for i in range(minBin, maxBin + 1):
        peaks.append((i, i * freqPerBin, i * freqPerBin * 120 / 4, hps[i]))
    peaks.sort(key=lambda x: x[3], reverse=True)
    return peaks[:3]

print("File 1 (5428) - True 484 RPM")
print("Linear Sum:", get_peaks('RPM-Testdateien/1-5428.mp3', True))
print("dB Sum:", get_peaks('RPM-Testdateien/1-5428.mp3', False))

print("\nFile 2 (5440) - True 969 RPM")
print("Linear Sum:", get_peaks('RPM-Testdateien/IMG_5440.MOV', True))
print("dB Sum:", get_peaks('RPM-Testdateien/IMG_5440.MOV', False))

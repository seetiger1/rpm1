import numpy as np
import librosa
import math

def get_peaks(filename, penalty_power):
    y, sr = librosa.load(filename, sr=44100)
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
            
    for i in range(1, length): hps[i] = hps[i] / math.pow(i, penalty_power)
    
    for i in range(length): hps[i] = 20 * math.log10(hps[i] + 1e-10)
    
    freqPerBin = (44100 / 2) / 4096
    minBin = max(1, math.ceil(15 / freqPerBin)) 
    maxBin = min(4095, math.ceil(500 / freqPerBin))
    
    peaks = []
    for i in range(minBin, maxBin + 1):
        peaks.append((i, i * freqPerBin, i * freqPerBin * 120 / 4, hps[i]))
    peaks.sort(key=lambda x: x[3], reverse=True)
    return peaks[:3]

print("File 1 (5428) - True 484 RPM")
print("Penalty 0.5:", get_peaks('RPM-Testdateien/1-5428.mp3', 0.5))
print("Penalty 0.75:", get_peaks('RPM-Testdateien/1-5428.mp3', 0.75))
print("Penalty 1.0:", get_peaks('RPM-Testdateien/1-5428.mp3', 1.0))

print("\nFile 2 (5440) - True 969 RPM")
print("Penalty 0.5:", get_peaks('RPM-Testdateien/IMG_5440.MOV', 0.5))
print("Penalty 0.75:", get_peaks('RPM-Testdateien/IMG_5440.MOV', 0.75))
print("Penalty 1.0:", get_peaks('RPM-Testdateien/IMG_5440.MOV', 1.0))

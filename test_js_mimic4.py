import numpy as np
import librosa
import math

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
hop = 4096

def run_test(penalty_exp):
    rpms = []
    for start_sample in range(len(y)//2, len(y) - 8192, hop*2):
        frame = y[start_sample:start_sample + 8192]
        window = np.blackman(8192)
        stft = np.fft.rfft(frame * window)
        spectrum = 20 * np.log10(np.abs(stft) + 1e-10)
        
        length = len(spectrum)
        hps = np.zeros(length)
        for i in range(length):
            hps[i] = math.pow(10, spectrum[i] / 20)
        
        for h in range(2, 7): # numHarmonics = 6
            for i in range(length):
                idx = i * h
                if idx < length:
                    hps[i] += math.pow(10, spectrum[idx] / 20)
        
        for i in range(1, length):
            hps[i] = hps[i] / math.pow(i, penalty_exp)
        
        for i in range(length):
            hps[i] = 20 * math.log10(hps[i] + 1e-10)
        
        freqPerBin = (44100 / 2) / 4096
        minBin = max(1, math.floor(15 / freqPerBin))
        maxBin = min(4095, math.ceil(500 / freqPerBin))
        
        maxVal = -float('inf')
        maxIndex = -1
        for i in range(minBin, maxBin + 1):
            if hps[i] > maxVal:
                maxVal = hps[i]
                maxIndex = i
                
        rpms.append(maxIndex * freqPerBin * 120 / 4)
    print(f"Penalty {penalty_exp}: Median {np.median(rpms):.0f} RPM (Mean: {np.mean(rpms):.0f}, Std: {np.std(rpms):.0f})")
    
run_test(0.5)
run_test(0.6)
run_test(0.7)
run_test(0.75)
run_test(0.8)
run_test(0.9)
run_test(1.0)

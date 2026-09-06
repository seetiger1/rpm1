import numpy as np
import librosa
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return b, a

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
b, a = butter_lowpass(150, sr)
y = lfilter(b, a, y)

hop = 4096
rpms = []
for i in range(0, len(y) - hop, hop):
    frame = y[i:i+hop]
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    min_period = int(sr / 100)
    max_period = int(sr / 12)
    
    if len(autocorr) > max_period:
        best_period = min_period + np.argmax(autocorr[min_period:max_period])
        best_freq = sr / best_period
        rpm = best_freq * 120 / 4
        rpms.append(rpm)
        if i % (hop*10) == 0:
            print(f"Time {i/sr:.1f}s: Freq = {best_freq:.2f} Hz, RPM = {rpm:.0f}")

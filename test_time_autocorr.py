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

start_sample = len(y) // 2
frame = y[start_sample:start_sample + 4096]

# time domain autocorrelation
autocorr = np.correlate(frame, frame, mode='full')
autocorr = autocorr[len(autocorr)//2:]

# Search for period between 16 Hz (2756 samples) and 100 Hz (441 samples)
min_period = int(sr / 100)
max_period = int(sr / 12)

best_period = min_period + np.argmax(autocorr[min_period:max_period])
best_freq = sr / best_period
print(f"Time Autocorr Peak: Period {best_period} -> {best_freq:.2f} Hz -> RPM: {best_freq * 120 / 4:.0f}")

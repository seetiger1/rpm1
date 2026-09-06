import numpy as np
import librosa

y, sr = librosa.load('RPM-Testdateien/1-5428.mp3', sr=44100)
start_sample = len(y) // 2
frame = y[start_sample:start_sample + 8192]

stft = librosa.stft(frame, n_fft=8192, center=False)
mag = np.abs(stft[:, 0])

# Spectral Autocorrelation
max_lag = 200 # about 100 Hz
autocorr = np.zeros(max_lag)

for lag in range(1, max_lag):
    # compute correlation
    corr = np.sum(mag[:-lag] * mag[lag:])
    autocorr[lag] = corr

# find peak in autocorr
freqs_per_bin = (sr / 2) / len(mag)

# We want RPM between 500 and 3000 -> 16 Hz and 100 Hz
min_lag = int(10 / freqs_per_bin)
max_lag_search = int(100 / freqs_per_bin)

best_lag = min_lag + np.argmax(autocorr[min_lag:max_lag_search])
best_freq = best_lag * freqs_per_bin
print(f"Spectral Autocorr Peak: Lag {best_lag} -> {best_freq:.2f} Hz -> RPM: {best_freq * 120 / 4:.0f}")


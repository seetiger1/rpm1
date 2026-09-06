import numpy as np
import librosa

y, sr = librosa.load('RPM-Testdateien/IMG_5440.MOV', sr=44100, duration=5.0)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=25, fmax=200, sr=sr)
valid_f0 = f0[voiced_flag]
if len(valid_f0) > 0:
    median_f0 = np.median(valid_f0)
    print(f"pYIN Median F0: {median_f0:.2f} Hz -> 4cyl: {median_f0 * 30:.0f} RPM, 6cyl: {median_f0 * 20:.0f} RPM")
else:
    print("pYIN found no valid F0.")

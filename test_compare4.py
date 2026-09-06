import numpy as np
import librosa
import math

def check_odd_harmonics(filename):
    y, sr = librosa.load(filename, sr=44100)
    hop = 4096
    start_sample = len(y)//2
    frame = y[start_sample:start_sample + 8192]
    window = np.blackman(8192)
    stft = np.fft.rfft(frame * window)
    mag = np.abs(stft)
    spectrum = 20 * np.log10(mag + 1e-10)
    spectrum = np.maximum(spectrum, 0)
    
    freqPerBin = (44100 / 2) / 4096
    bin_48_45 = round(48.45 / freqPerBin)
    bin_80_75 = round(80.75 / freqPerBin)
    bin_32_30 = round(32.30 / freqPerBin)
    bin_64_60 = round(64.60 / freqPerBin)
    
    print(f"32.30 Hz (Even): {spectrum[bin_32_30]:.2f} dB")
    print(f"48.45 Hz (Odd) : {spectrum[bin_48_45]:.2f} dB")
    print(f"64.60 Hz (Even): {spectrum[bin_64_60]:.2f} dB")
    print(f"80.75 Hz (Odd) : {spectrum[bin_80_75]:.2f} dB")

print("File 1 (5428) - True 484 RPM")
check_odd_harmonics('RPM-Testdateien/1-5428.mp3')

print("\nFile 2 (5440) - True 969 RPM")
check_odd_harmonics('RPM-Testdateien/IMG_5440.MOV')

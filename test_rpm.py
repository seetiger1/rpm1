import os
import sys
import numpy as np
import librosa
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def harmonic_product_spectrum(spectrum_db, num_harmonics=3):
    length = len(spectrum_db)
    hps = np.copy(spectrum_db)
    hps = np.maximum(hps, -150)
    
    for h in range(2, num_harmonics + 1):
        for i in range(length):
            idx = i * h
            if idx < length:
                hps[i] += max(spectrum_db[idx], -150)
            else:
                hps[i] = -450
    return hps

def find_peak_frequency(hps, sample_rate, fft_size, min_freq=15, max_freq=500):
    bin_count = len(hps)
    freq_per_bin = (sample_rate / 2) / bin_count
    
    min_bin = max(1, int(np.floor(min_freq / freq_per_bin)))
    max_bin = min(bin_count - 1, int(np.ceil(max_freq / freq_per_bin)))
    
    if min_bin > max_bin:
        return 0, -1, 0

    search_range = hps[min_bin:max_bin+1]
    max_val = np.max(search_range)
    max_index_rel = np.argmax(search_range)
    max_index = min_bin + max_index_rel
    
    # Noise gate
    if max_val < -350:
        return 0, -1, 0
        
    mean_val = np.mean(search_range)
    if max_val - mean_val < 15:
        return 0, -1, 0
        
    # Parabolic interpolation
    peak_freq = max_index * freq_per_bin
    if min_bin < max_index < max_bin:
        alpha = hps[max_index - 1]
        beta = hps[max_index]
        gamma = hps[max_index + 1]
        denom = alpha - 2 * beta + gamma
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
            peak_freq = (max_index + p) * freq_per_bin
            
    return peak_freq, max_index, max_val

def analyze_audio(file_path):
    print(f"Analyzing {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=44100)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
        
    y_filtered = lowpass_filter(y, 150, sr)
    
    fft_size = 8192
    hop_length = 4096
    
    stft = librosa.stft(y_filtered, n_fft=fft_size, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)
    spectrum_db = 20 * np.log10(magnitude + 1e-10)
    
    rpms = []
    
    for i in range(spectrum_db.shape[1]):
        frame_db = spectrum_db[:, i]
        hps = harmonic_product_spectrum(frame_db, 3)
        peak_freq, idx, mag = find_peak_frequency(hps, sr, fft_size)
        
        rpm = 0
        if peak_freq > 0:
            rpm = round((peak_freq * 120) / 4)
            
        rpms.append(rpm)
        
        time_sec = i * hop_length / sr
        if i % 20 == 0:
            print(f"Time {time_sec:.1f}s: Freq = {peak_freq:.1f} Hz, RPM = {rpm}")
            
    half_idx = len(rpms) // 2
    second_half = rpms[half_idx:]
    second_half_valid = [r for r in second_half if r > 0]
    if second_half_valid:
        print(f"Median RPM in second half: {np.median(second_half_valid)}")
    else:
        print("No valid RPMs found in second half.")

if __name__ == '__main__':
    analyze_audio('RPM-Testdateien/1-5428.mp3')

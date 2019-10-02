#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
import matplotlib.pyplot as plt

# Get sample signal and reference from records
sample_record = records[15]
sample_signal = sample_record.ppg[1]
sample_peaks = np.array(sample_record.hrv[0]) - sample_record.ppg[0][0] 

# Apply detector
detector = crossover_detector()
detector.set_parameters(alpha_fast = 0.5144310567075446, alpha_slow = 0.9640168863482605)                   # Current best set (01/10/19), with cost of 0.36697370
fast_averages, slow_averages, crossover_indices, detected_peaks = detector.detect_peaks(sample_signal)

# Plot signal and reference
plt.figure()
plt.title("PPG peak detection")
plt.plot(sample_signal, color='k', label="PPG signal")
plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")

# Plot detector's output
#plt.plot(fast_averages, color='magenta', label="fast average")
#plt.plot(slow_averages, color='navy', label="slow average")
#plt.plot(crossover_indices, color='green', label="crossover index")
plt.plot(detected_peaks, color='gray', label="Detected peaks")
plt.legend()

plt.show()
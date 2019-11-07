#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
import matplotlib.pyplot as plt

# Get sample signal and reference from records
sample_record = records[10]
sample_signal = sample_record.ppg[1]
sample_peaks = np.array(sample_record.beats[0]) - sample_record.ppg[0][0] 

# Apply detector
detector = crossover_detector()
#detector.set_parameters(alpha_fast = 0.8329186816539544, alpha_slow = 0.9463972836760943)                   # Current best set (17/10/19), with cost of 0.4324561691187358
#detector.set_parameters(alpha_fast = 0.8801246553858418, alpha_slow = 0.8862226689490318)                   # Current best set of husm (07/11/19), with cost of 0.95
#fast_averages, slow_averages, crossover_indices, detected_peaks = detector.detect_peaks(sample_signal)
#confusion_matrix = detector.signal_confusion_matrix(detected_peaks, sample_peaks)[:-1]
#print('Record confusion matrix: [TP,TN,FP,FN]' + str(confusion_matrix))

# Plot signal and reference
# plt.figure()
# plt.title("PPG peak detection")
# plt.plot(sample_signal, color='k', label="PPG signal")
# plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")

# # Plot detector's output
# #plt.plot(fast_averages, color='magenta', label="fast average")
# #plt.plot(slow_averages, color='navy', label="slow average")
# #plt.plot(crossover_indices, color='green', label="crossover index")
# plt.plot(detected_peaks, color='gray', label="Detected peaks")
# plt.legend()

# plt.show()

detector.set_parameters_var(var_alpha = 0.1)
variances, averages = detector.detect_peaks_var(sample_signal);
# Plot signal and reference
plt.figure()
plt.title("PPG average and variance")
plt.plot(sample_signal, color='k', label="PPG signal")
plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")
plt.plot(averages, color='magenta', label="average")
plt.plot(variances, color='navy', label="variance")
plt.legend()
plt.show()
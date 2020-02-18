#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
import matplotlib.pyplot as plt
import pickle as pkl
import sys

if len(sys.argv) != 2:
    print("Please enter record number (0:60)")
    exit(-1)
    
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
    
# Get sample signal and reference from records
record_number = int(sys.argv[1])
sample_record = records[record_number]
sample_signal = sample_record.ppg[1]
sample_peaks = np.array(sample_record.beats[0]) - sample_record.ppg[0][0] 

detector = crossover_detector()

# Apply crossover detector
#detector.set_parameters(alpha_fast = 0.8329186816539544, alpha_slow = 0.9463972836760943)                   # Current best set (17/10/19), with cost of 0.4324561691187358
#detector.set_parameters_cross(alpha_fast = 0.9684424413846544, alpha_slow = 0.9689074349612342)                    # Current best set of husm (07/11/19), with cost of 0.95
#detector.set_parameters_cross(alpha_fast = 0.9684424413846544, alpha_slow = 0.9689074349612342, percentage_threshold = 0.8)                    # Current best set of husm (07/11/19), with cost of 0.95
detector.set_parameters_cross(alpha_fast = 0.9194304850895123 , alpha_slow = 0.9387829409026597, percentage_threshold = 0.35405603150462084)                    # Current best set of husm (21/01/20), with cost of 0.9

fast_averages, slow_averages, crossover_indices, detected_peaks = detector.detect_peaks_cross(sample_signal)
detected_peaks = detector.ignore_short_peaks(detected_peaks, 13)
detected_positions = detector.peak_positions(sample_signal, detected_peaks)
confusion_matrix = detector.signal_confusion_matrix(detected_peaks, sample_peaks)[:-1]
print('Record confusion matrix: [TP,TN,FP,FN]' + str(confusion_matrix))

lit_cm = detector.literature_signal_confusion_matrix(detected_positions, sample_peaks)
print('Record literature confusion matrix: [TP,FP,FN]' + str(lit_cm))

#Plot signal and reference
plt.figure()
plt.title("PPG peak detection")
plt.plot(sample_signal, color='k', label="PPG signal")
plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")
plt.scatter(detected_positions, [1]*len(detected_positions), label="Found peaks")

# Plot detector's output
plt.plot(fast_averages, color='magenta', label="fast average")
plt.plot(slow_averages, color='navy', label="slow average")
plt.plot(crossover_indices, color='green', label="crossover index")
plt.plot(detected_peaks, color='gray', label="Detected peaks")
plt.legend()

plt.show()

## Apply variance detector
# detector.set_parameters_var(var_alpha = 0.002, avg_alpha = 0.9, var_threshold = 100.0)
# variances, averages, detected_peaks = detector.detect_peaks_var(sample_signal);
# # Plot signal and reference
# plt.figure()
# plt.title("PPG average and variance")
# plt.plot(sample_signal, color='k', label="PPG signal")
# plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")
# plt.plot(detected_peaks, color = 'y', label = "Detected peaks")
# plt.plot(averages, color='magenta', label="average")
# plt.plot(variances, color='navy', label="variance")

# plt.legend()
# plt.show()

## Apply Mixed detector
# detector.set_parameters_mix(alpha_fast = 0.9656528722478156 , alpha_slow = 0.9668991435014739, 
                            # var_alpha = 0.8322917528906034, avg_alpha = 0.18278088860663844, var_threshold = 21.16502812624924)
# detected_peaks = detector.detect_peaks_mix(sample_signal);

# Apply bagging
#solution_archive = pkl.load(open("solution_archive.data","rb"))
#alphas_fast = solution_archive[:, 0]
#alphas_slow = solution_archive[:, 1]
#detected_peaks = detector.bagging_join_detections(alphas_fast, alphas_slow, sample_signal)


# Plot signal and reference
# plt.figure()
# plt.title("PPG average and variance")
# plt.plot(sample_signal, color='k', label="PPG signal")
# plt.scatter(sample_peaks, [1]*len(sample_peaks), label="Reference peaks")
# plt.plot(detected_peaks, color = 'y', label = "Detected peaks")
#plt.plot(averages, color='magenta', label="average")
#plt.plot(variances, color='navy', label="variance")

# plt.legend()
# plt.show()
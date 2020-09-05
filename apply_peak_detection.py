#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
import matplotlib.pyplot as plt
import pickle as pkl
import sys
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz

if len(sys.argv) != 3:
    print('Please enter the first and last record numbers')
    exit(-1)
    
first_rec = int(sys.argv[1])
last_rec = int(sys.argv[2])

if first_rec > last_rec:
    print('Error, last rec must be greater than first rec')
    exit(-1)
    
# Define crossover detector
detector = crossover_detector()
detector.set_parameters_cross(alpha_crossover = 0.8705192717851324, alpha_fast = 0.903170529094925  , alpha_slow = 0.9586798163470798)                    

# Get sample signal and reference from records
#record_number = int(sys.argv[1])
for record_number in range(first_rec, last_rec + 1):

    sample_record = records[record_number]
    sample_signal = sample_record.ppg[1]
    sample_peaks = np.array(sample_record.beats[0]) - sample_record.ppg[0][0] 

    fast_averages, slow_averages, crossover_indices, detected_peaks = detector.detect_peaks_cross(sample_signal)
    #detected_peaks = detector.ignore_short_peaks(detected_peaks, 20)
    detected_positions = detector.peak_positions(sample_signal, detected_peaks)
    
    # confusion_matrix = detector.signal_confusion_matrix(detected_peaks, sample_peaks)[:-1]
    # print('Record confusion matrix: [TP,TN,FP,FN]' + str(confusion_matrix))

    lit_cm = detector.literature_signal_confusion_matrix(detected_positions, sample_peaks)
    print('\nRecord ' + str(record_number) + ' literature confusion matrix: [TP,FP,FN]' + str(lit_cm))

    #Plot signal and reference
    plt.figure()
    plt.title('PPG peak detection (rec ' + str(record_number) + ')')
    plt.plot(sample_signal, color='k', label='PPG signal')
    plt.scatter(sample_peaks, [0.3]*len(sample_peaks), label='Reference peaks')
    plt.scatter(detected_positions, [0.3]*len(detected_positions), label='Found peaks')

    # Plot detector's output
    plt.plot(fast_averages, color='magenta', label='fast average')
    plt.plot(slow_averages, color='navy', label='slow average')
    plt.plot(crossover_indices, color='green', label='crossover index')
    plt.plot(0.3*np.array(detected_peaks), color='gray', label='Detected peaks')
    plt.legend()

plt.show()

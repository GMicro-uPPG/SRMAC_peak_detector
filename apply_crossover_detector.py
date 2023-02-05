#!python3

# MIT License

# Copyright (c) 2021 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Victor O. Costa

# Python std library
import sys
if len(sys.argv) != 3:
    print('Please enter the first and last record indices')
    exit(-1)
# Application modules
from crossover_detector import crossover_detector
from read_datasets import records
import utilities
# Third party
import numpy as np
import matplotlib.pyplot as plt

first_rec = int(sys.argv[1])
last_rec = int(sys.argv[2])
  
if first_rec > last_rec:
    print('Error, last record must be greater than first record')
    exit(-1)
if last_rec > len(records) - 1 or last_rec < 0 or first_rec < 0:
    print(f'Error, record index must be in the range ]0,{len(records)-1}]')
    exit(-1)

# Define crossover detector
Fs = 200
# detector = crossover_detector(0.8705, 0.9032, 0.9387, 0)
detector = crossover_detector(0.9592, 0.8991, 0.9324, 5.710e-05)

accumulated_cm = [0, 0, 0]
# Get sample signal and reference from records
for record_number in range(first_rec, last_rec + 1):

    sample_record = records[record_number]
    sample_signal = sample_record.ppg[1]
    reference_peaks = np.array(sample_record.beats[0]) - sample_record.ppg[0][0] 

    fast_averages, slow_averages, crossover_indices, peak_blocks, peak_positions = detector.get_peak_results(sample_signal, Fs)
    
    lit_cm = utilities.signal_confusion_matrix(peak_positions, reference_peaks, Fs)
    accumulated_cm = np.array(accumulated_cm) + np.array(lit_cm)

    print('\nRecord ' + str(record_number) + ' literature confusion matrix: [TP,FP,FN]' + str(lit_cm))
    print('Number of reference peaks: ' + str(len(reference_peaks)))
    print('Number of peaks found: ' + str(len(peak_positions)))
    
    order = 2
    low_cut = 0.5   # Hz
    high_cut = 8    # Hz 
    filtered_ppg = utilities.biquad_butter_bandpass(sample_signal, order, low_cut, high_cut, Fs)
    
    #Plot signal and reference
    plt.figure()
    plt.title('PPG peak detection (rec ' + str(record_number) + ')')
    plt.plot(sample_signal, color='k', label='PPG signal')

    plt.scatter(reference_peaks, [0.3]*len(reference_peaks), label='Reference peaks', linewidth=4.5)
    plt.scatter(peak_positions, [0.3]*len(peak_positions), label='Found peaks', linewidth=1.5)

    # Plot filtered signal
    plt.plot(filtered_ppg, color='k', linewidth=3.5)

    # Plot detector's output
    # plt.plot(fast_averages, color='magenta', label='$MA_{fast}$', linewidth=2)
    # plt.plot(slow_averages, color='navy', label='$MA_{slow}$', linewidth=2)
    plt.plot(1 * np.array(crossover_indices), color='green', label='crossover index', linewidth=2)
    plt.plot(0.3*np.array(peak_blocks), color='gray', label='Detected peaks')
    plt.legend(loc='upper right', fontsize=16)

print('Accumulated confusion matrix: [TP,FP,FN]' + str(accumulated_cm))
plt.show()

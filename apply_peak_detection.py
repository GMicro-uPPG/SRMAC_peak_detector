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
    print('Please enter the first and last record numbers')
    exit(-1)
# Own
from ppg_peak_detection import crossover_detector
from read_datasets import records # This will load 66 records. Rercord sample rate = 200 Hz
import optimization_utilities

# Third party
import numpy as np
import matplotlib.pyplot as plt

first_rec = int(sys.argv[1])
last_rec = int(sys.argv[2])

if first_rec > last_rec:
    print('Error, last rec must be greater than first rec')
    exit(-1)

# Define crossover detector
detector = crossover_detector(0.8705192717851324, 0.903170529094925, 0.9586798163470798, 200)              

# Get sample signal and reference from records
for record_number in range(first_rec, last_rec + 1):

    sample_record = records[record_number]
    sample_signal = sample_record.ppg[1]
    sample_peaks = np.array(sample_record.beats[0]) - sample_record.ppg[0][0] 

    fast_averages, slow_averages, crossover_indices, detected_peaks = detector.get_peak_blocks(sample_signal)
    detected_positions = optimization_utilities.peak_positions(sample_signal, detected_peaks)

    lit_cm = optimization_utilities.signal_confusion_matrix(detected_positions, sample_peaks, 200)
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

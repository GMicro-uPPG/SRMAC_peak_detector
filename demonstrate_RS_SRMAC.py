#!python3

# MIT License

# Copyright (c) 2023 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

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

# Application modules
from SRMAC_detector import SRMAC_detector
import read_datasets
import utilities
import optimization
# Third party
import numpy as np
import time_manager

print('\nFirst timestamp: ' + str(time_manager.time.getTimestamp()))
print('First time: ' + str(time_manager.time.getTime()))		

# Load reference data (44 records for training and 22 for testing)
# Test data is composed of an equal number of healty and dpoc records
records = read_datasets.getHUSMppg()	
train_records = records[11:-11]
print('\nTrain records: [11:-11], len = ' + str(len(train_records)))
test_records = records[0:11] + records[-11:]
print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

# Sampling frequency
Fs = 200
# Number of runs to extract stats from
num_runs = 3
print('\nNumber of runs = ' + str(num_runs))
# Iterations of interest for random search
iterations_of_interest = [1, 5]                                       
print('Iterations of interest = ' + str(iterations_of_interest))

verbosity = True
# Train acc histories
hist_train_accs = []
hist_test_accs = []
for _ in range(num_runs):
		# Get history of solutions defined by iterations of interest
		solutions_of_interest = optimization.random_search_SRMAC(train_records = train_records,
														iterations_of_interest = iterations_of_interest,
														alpha_min = 0.7, alpha_max = 1,
														thr_min = 0, thr_max = 5e-4,
														sampling_frequency=Fs, verbosity=verbosity)
		run_train_accuracies = []
		run_test_accuracies = []
		# For each solution define a model and extract test acc
		for soi in solutions_of_interest:
				alpha_cross, alpha_fast, alpha_slow, threshold, train_cost = soi
				peak_detector = SRMAC_detector(alpha_cross, alpha_fast, alpha_slow, threshold)
				# Train
				train_accuracy = 1 - train_cost     # Train cost is 1 - acc
				run_train_accuracies.append(train_accuracy)
				# Test
				test_cm = utilities.record_set_confusion_matrix(peak_detector, test_records, Fs)
				test_precision = test_cm[0] / (test_cm[0] + test_cm[1])
				test_recall =    test_cm[0] / (test_cm[0] + test_cm[2])
				test_accuracy = (test_precision + test_recall)/2
				run_test_accuracies.append(test_accuracy)
		hist_train_accs.append(list(run_train_accuracies))
		hist_test_accs.append(list(run_test_accuracies))

hist_train_accs = np.array(hist_train_accs)
hist_test_accs = np.array(hist_test_accs)

print('\nTrain accuracies through runs')
print(hist_train_accs)
print('\nTest accuracies  through runs')
print(hist_test_accs)
print(f'\nTrain acc: {np.mean(hist_train_accs[:,-1])} ({np.std(hist_train_accs[:,-1])})')
print(f'Test acc:  {np.mean(hist_test_accs[:,-1])} ({np.std(hist_test_accs[:,-1])})')

print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
print('Last time: ' + str(time_manager.time.getTime()))
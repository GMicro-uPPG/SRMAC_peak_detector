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

# Application modules
from TERMA_detector import TERMA_detector
from read_datasets import records # This will load 66 records. Rercord sample rate = 200 Hz
import utilities
import optimization
# Third party
import numpy as np
import time_manager

# Load reference data (44 records for training and 22 for testing)
# Test data is composed of an equal number of healty and dpoc records
if len(records) != 66:
    print('Number of records is not 66')
    exit(-1)
    
train_records = records[11:-11]
print('Train records: [11:-11], len = ' + str(len(train_records)))
test_records = records[0:11] + records[-11:]
print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

# Sampling frequency
Fs = 200
# Lists of parameters for GS
W1 = [54, 111]
W2 = [545, 694]
beta = [0.5]

verbosity = True
# Get best solution from grid search over train records
best_sol = optimization.grid_search_TERMA(train_records = train_records, W1_list = W1, W2_list = W2,
                                    beta_list = beta, sampling_frequency = Fs, verbosity = verbosity)

detector = TERMA_detector(best_sol[0], best_sol[1], best_sol[2])
train_acc = 1 - best_sol[-1]

test_cm = utilities.record_set_confusion_matrix(detector, test_records, Fs)
test_precision = test_cm[0] / (test_cm[0] + test_cm[1])
test_recall =    test_cm[0] / (test_cm[0] + test_cm[2])
test_acc = (test_precision + test_recall)/2
print(f'Train acc: {train_acc} \nTest acc: {test_acc}')
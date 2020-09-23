#!python3

# MIT License

# Copyright (c) 2016 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

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

# Python std library
import sys
# Own
from ppg_peak_detection import crossover_detector
from ppg_peak_detection import random_search_crossover
# Third party
import numpy as np

## Sanity check
# Number of cross-validation folds
if len(sys.argv) != 2:
    print('Error, pass the number of cross-validation folds to the script')
    exit(-1)
num_folds = int(sys.argv[1])
if num_folds <= 0 or num_folds > 66:
    print('Error, there are 66 records')
    exit(-1)


# Load records (PPG signals and peak references)
from read_datasets import records # This will load 60 records (o to 59). Record sample rate = 200 Hz
num_recs = len(records)
print(f'Loaded {num_recs} records')
if num_recs == 0:
    exit(-1)

# Length of the cross-validation folds
fold_len = num_recs // num_folds
print(f'The size of each fold is {fold_len} records')
leftovers = num_recs % num_folds
if leftovers > 0:
    print(f'There are {leftovers} unused records')
    print(f'This wont happen if num_folds is a divisor of {num_recs}')

# Sampling frequency
Fs = 200
# Number of random search iterations per run, and runs per fold
num_iterations = 100   
num_runs = 30                                    
verbosity = False
print(f'RS iterations per run = {num_iterations}\nRS runs per folds = {num_runs}')

# Generate num_folds splits of the loaded record and run random search a number of times for each split
cv_accuracies = []
for fold_i in range(num_folds):
    # Split the record
    fold_validation = records[fold_i * fold_len : (fold_i + 1) * fold_len]
    fold_train      = records[0 : fold_i * fold_len] + records[(fold_i + 1) * fold_len : len(records) - leftovers]
    # print(f'Val: {len(fold_validation)}; Train: {len(fold_train)}')
    
    # Run random search a number of times and report validation accuracies
    validation_accuracies   = []
    for run in range(num_runs):
        rs_solution = random_search_crossover(train_records = fold_train, num_iterations = num_iterations, min_alpha = 0.7, max_alpha = 1, sampling_frequency=Fs, verbosity=verbosity)
        alpha_cross, alpha_fast, alpha_slow, train_cost = rs_solution
        peak_detector = crossover_detector(alpha_cross, alpha_fast, alpha_slow, Fs)
    
        # Here accuracy is the average between precision and recall
        # Train
        train_accuracy = 1 - train_cost     # Train cost is 1 - train_acc
        print(f'Run {run}')
        print(f'Train acc = {train_accuracy}')
        
        # Validation accuracy
        validation_conf_mat = peak_detector.literature_record_set_confusion_matrix(fold_validation)
        tp, fp, fn = validation_conf_mat
        val_precision = tp / (tp + fp)
        val_recall    = tp / (tp + fn)
        val_accuracy  = (val_precision + val_recall)/2
        validation_accuracies.append(val_accuracy)
        print(f'Validation acc = {val_accuracy}')

    
    fold_accuracy = np.mean(validation_accuracies)
    cv_accuracies.append(fold_accuracy)
    print(f'Fold accs len = {len(validation_accuracies)}')
    print(f'Fold avg accuracy = {fold_accuracy}')
    
# print(cv_accuracies)
mean_acc = np.mean(cv_accuracies)
print(f'Outter accs len = {len(cv_accuracies)}')
print(f'Outter avg accuracy = {mean_acc}')

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
# Own
from TERMA_detector import TERMA_detector
import utilities
import optimization
# Third party
import numpy as np

## Sanity check
# Number of cross-validation folds
if len(sys.argv) != 2:
    print('Error, inform in how many cross-validation folds the records will be split')
    exit(-1)
num_folds = int(sys.argv[1])

if num_folds <= 0 or num_folds > 66:
    print('Error, the number of folds should be in the range [1,66]')
    exit(-1)

# Load records (PPG signals and peak references)
from read_datasets import records   # This import will load 66 records. Record sample rate = 200 Hz

# Length of the cross-validation folds
num_recs = len(records)
fold_len = num_recs // num_folds
print(f'The size of each fold is {fold_len} records')
leftovers = num_recs % num_folds
if leftovers > 0:
    print(f'There are {leftovers} unused records')
    print(f'This wont happen if {num_recs} is divisible by the number of CV folds')

# Sampling frequency
Fs = 200
# Number of random search iterations per run, and runs per fold
iterations_of_interest = [50, 100, 150, 200, 250, 300]
num_runs = 30
verbosity = True
print(f'RS iterations of interest = {iterations_of_interest}\nRS runs per folds = {num_runs}')

# Generate num_folds splits of the loaded record and run random search a number of times for each split
cv_parameters = []
cv_precisions = []
cv_recalls = []

for fold_i in range(num_folds):
    # Split the record
    fold_validation = records[fold_i * fold_len : (fold_i + 1) * fold_len]
    fold_train      = records[0 : fold_i * fold_len] + records[(fold_i + 1) * fold_len : len(records) - leftovers]
    
    # Store validation precision and recall for each run and for all iterations of interest
    fold_parameter_history = []
    fold_precision_history = []
    fold_recall_history = []
    # Run random search a number of times
    for run in range(num_runs):
        # Get history of solutions defined by iterations of interest
        solutions_of_interest = optimization.random_search_TERMA(train_records = fold_train, iterations_of_interest = iterations_of_interest,
                                                                 W1_min = 54, W1_max = 111,
                                                                 W2_min = 545,W2_max = 694,
                                                                 beta_min = 0, beta_max = 0.1, 
                                                                 sampling_frequency=Fs, verbosity=verbosity)
        # Parameters found, and also precisions and recalls of interest for this run
        run_parameter_sets = []
        run_precisions = []
        run_recalls = []
        # For each solution define a model and extract validation precision, recall and accuracy
        for soi in solutions_of_interest:
            W1, W2, beta, train_cost = soi
            peak_detector = TERMA_detector(W1, W2, beta)

            # Validation triangular confusion matrix
            validation_conf_mat = utilities.record_set_confusion_matrix(peak_detector, fold_validation, Fs)
            tp, fp, fn = validation_conf_mat

            if tp != 0:
                val_precision = tp / (tp + fp)
                val_recall    = tp / (tp + fn)
            else:
                val_precision = 0
                val_recall = 0

            # Keep parameters, precisions and recalls for this run
            run_parameter_sets.append(list(soi[:-1]))
            run_precisions.append(val_precision)
            run_recalls.append(val_recall)

        # Verify dimensions (1)
        print('Run check')
        print(f'{np.shape(run_parameter_sets)} .. should be ({len(iterations_of_interest)},3)')
        print(f'{np.shape(run_precisions)} .... should be ({len(iterations_of_interest)})')
        print(f'{np.shape(run_recalls)} .... should be ({len(iterations_of_interest)})')

        # Keep data of each run
        fold_parameter_history.append(list(run_parameter_sets))
        fold_precision_history.append(list(run_precisions))
        fold_recall_history.append(list(run_recalls))

    # Verify dimensions (2)
    print('Fold check')
    print(f'{np.shape(fold_parameter_history)} .. should be ({num_runs},{len(iterations_of_interest)},3)')
    print(f'{np.shape(fold_precision_history)} .... should be ({num_runs},{len(iterations_of_interest)})')
    print(f'{np.shape(fold_recall_history)} .... should be ({num_runs},{len(iterations_of_interest)})')

    # Keep data of each fold
    cv_parameters.append(list(fold_parameter_history))
    cv_precisions.append(list(fold_precision_history))
    cv_recalls.append(list(fold_recall_history))

# Verify dimensions (3)
print('Full CV check')
print(f'{np.shape(cv_parameters)} .. should be ({num_folds},{num_runs},{len(iterations_of_interest)},3)')
print(f'{np.shape(cv_precisions)} .... should be ({num_folds},{num_runs},{len(iterations_of_interest)})')
print(f'{np.shape(cv_recalls)}    .... should be ({num_folds},{num_runs},{len(iterations_of_interest)})')

if num_folds == 22:
    base_filename = f'search_results/LOSOCV_RS_TERMA_{num_folds}folds_{num_runs}runs_'
else:
    base_filename = f'search_results/CV_RS_TERMA_{num_folds}folds_{num_runs}runs_'
		
np.save(base_filename + 'parameters.npy', cv_parameters)
np.save(base_filename + 'precisions.npy', cv_precisions)
np.save(base_filename + 'recalls.npy',    cv_recalls)
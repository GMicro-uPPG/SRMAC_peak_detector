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
iterations_of_interest = [i*50 for i in range(1,21)]
num_runs = 30
verbosity = False
print(f'RS iterations of interest = {iterations_of_interest}\nRS runs per folds = {num_runs}')

# Generate num_folds splits of the loaded record and run random search a number of times for each split
cv_parameters = []
cv_precisions = []
cv_recalls = []
for fold_i in range(num_folds):
    # Split the record
    fold_validation = records[fold_i * fold_len : (fold_i + 1) * fold_len]
    fold_train      = records[0 : fold_i * fold_len] + records[(fold_i + 1) * fold_len : len(records) - leftovers]
    # print(f'Val: {len(fold_validation)}; Train: {len(fold_train)}')
    
    # Store validation precision and recall for each run and for all iterations of interest
    fold_parameter_history = []
    fold_precision_history = []
    fold_recall_history = []
    # Run random search a number of times
    for run in range(num_runs):
        # Get history of solutions defined by iterations of interest
        solutions_of_interest = random_search_crossover(train_records = fold_train, iterations_of_interest = iterations_of_interest,
                                                        min_alpha = 0.7, max_alpha = 1, sampling_frequency=Fs, verbosity=verbosity)
        # Parameters found, and also precisions and recalls of interest for this run
        run_parameter_sets = []
        run_precisions = []
        run_recalls = []
        # For each solution define a model and extract validation precision, recall and accuracy
        for soi in solutions_of_interest:
            alpha_cross, alpha_fast, alpha_slow, train_cost = soi
            peak_detector = crossover_detector(alpha_cross, alpha_fast, alpha_slow, Fs)
            
            # Validation triangular confusion matrix
            validation_conf_mat = peak_detector.literature_record_set_confusion_matrix(fold_validation)
            tp, fp, fn = validation_conf_mat
            val_precision = tp / (tp + fp)
            val_recall    = tp / (tp + fn)
            
            # Keep parameters, precisions and recalls for this run
            run_parameter_sets.append(list(soi[:-1]))
            run_precisions.append(val_precision)
            run_recalls.append(val_recall)
            
            # Compute acc to validate code
            # val_accuracy  = (val_precision + val_recall)/2          # Here accuracy is the average between precision and recall
            # validation_accuracies.append(val_accuracy)
            # print(f'Validation acc = {val_accuracy}')
        
        # Validation 1
        # run_accs = (np.array(run_precisions) + np.array(run_recalls))/2
        # print('run accs')
        # print(run_accs)
        
        # Validation 2
        print('Run check')
        print(f'{np.shape(run_parameter_sets)} .. should be ({len(iterations_of_interest)},3)')
        print(f'{np.shape(run_precisions)} .... should be ({len(iterations_of_interest)})')
        print(f'{np.shape(run_recalls)} .... should be ({len(iterations_of_interest)})')
        
        # Keep data of each run
        fold_parameter_history.append(list(run_parameter_sets))
        fold_precision_history.append(list(run_precisions))
        fold_recall_history.append(list(run_recalls))
    
    # Validation 3
    print('Fold check')
    print(f'{np.shape(fold_parameter_history)} .. should be ({num_runs},{len(iterations_of_interest)},3)')
    print(f'{np.shape(fold_precision_history)} .... should be ({num_runs},{len(iterations_of_interest)})')
    print(f'{np.shape(fold_recall_history)} .... should be ({num_runs},{len(iterations_of_interest)})')
    
    # Keep data of each fold
    cv_parameters.append(list(fold_parameter_history))
    cv_precisions.append(list(fold_precision_history))
    cv_recalls.append(list(fold_recall_history))

# Validation 4
print('Full CV check')
print(f'{np.shape(cv_parameters)} .. should be ({num_folds},{num_runs},{len(iterations_of_interest)},3)')
print(f'{np.shape(cv_precisions)} .... should be ({num_folds},{num_runs},{len(iterations_of_interest)})')
print(f'{np.shape(cv_recalls)}    .... should be ({num_folds},{num_runs},{len(iterations_of_interest)})')

if num_folds == 22:
    base_filename = f'LOSOCV_{num_folds}folds_{num_runs}runs_'
else:
    base_filename = f'CV_{num_folds}folds_{num_runs}runs_'
np.save(base_filename + 'parameters.npy', cv_parameters)
np.save(base_filename + 'precisions.npy', cv_precisions)
np.save(base_filename + 'recalls.npy',    cv_recalls)

# fold_accuracy = np.mean(validation_accuracies)
# cv_accuracies.append(fold_accuracy)
# print(f'Fold accs len = {len(validation_accuracies)}')
# print(f'Fold avg accuracy = {fold_accuracy}')
# print(cv_accuracies)
# mean_acc = np.mean(cv_accuracies)
# print(f'Outter accs len = {len(cv_accuracies)}')
# print(f'Outter avg accuracy = {mean_acc}')

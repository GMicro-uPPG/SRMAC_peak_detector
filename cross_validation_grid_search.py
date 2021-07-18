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
    print('Error, pass the number of cross-validation folds to the script')
    exit(-1)
num_folds = int(sys.argv[1])
if num_folds <= 0 or num_folds > 66:
    print('Error, there are 66 records')
    exit(-1)

# Load records (PPG signals and peak references)
from read_datasets import records   # This import will load 66 records. Record sample rate = 200 Hz
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
    print(f'This wont happen if {num_recs} is divisible by num_folds')

# Sampling frequency
Fs = 200
# Lists of parameters for GS
W1_list = [54, 111]
W2_list = [545, 694]
beta_list = [0.1, 1]
verbosity = False

# Generate num_folds splits of the loaded record and run random search a number of times for each split
cv_parameters = []
cv_precisions = []
cv_recalls = []
for fold_i in range(num_folds):
    # Split the record
    fold_validation = records[fold_i * fold_len : (fold_i + 1) * fold_len]
    fold_train      = records[0 : fold_i * fold_len] + records[(fold_i + 1) * fold_len : len(records) - leftovers]
    # print(f'Val: {len(fold_validation)}; Train: {len(fold_train)}')
    
    # Search for TERMA parameters with grid search
    found_solution = optimization.grid_search_TERMA(train_records = fold_train,
                     W1_list = W1_list, W2_list = W2_list, beta_list = beta_list,
                     sampling_frequency = Fs, verbosity = verbosity)
    
    # Define TERMA model with the found parameters and compute validation metrics
    W1, W2, beta, _ = found_solution
    detector = TERMA_detector(W1, W2, beta)              
    tp, fp, fn = utilities.record_set_confusion_matrix(detector, fold_validation, Fs)
    val_precision = tp / (tp + fp)
    val_recall    = tp / (tp + fn)
    
    # Store parameters found by GS and metrics for the validation fold
    cv_parameters.append( list(found_solution) )
    cv_precisions.append(val_precision)
    cv_recalls.append(val_recall)
    
print(f'Parameters { {np.size(cv_parameters) }}')
print(cv_parameters)
print(f'Parameters { {np.size(cv_precisions) }}')
print(cv_precisions)
print(f'Parameters { {np.size(cv_recalls) }}')
print(cv_recalls)

# Save results in binary files
if num_folds == 22:
    base_filename = f'TERMA_GS_LOSOCV_{num_folds}folds_'
else:
    base_filename = f'TERMA_GS_CV_{num_folds}folds_'
np.save(base_filename + 'parameters.npy', cv_parameters)
np.save(base_filename + 'precisions.npy', cv_precisions)
np.save(base_filename + 'recalls.npy',    cv_recalls)
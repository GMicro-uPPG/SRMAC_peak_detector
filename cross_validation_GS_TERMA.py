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
import time
# Own
from TERMA_detector import TERMA_detector
import utilities
import optimization
import read_datasets
# Third party
import numpy as np

## Sanity check
# Number of cross-validation folds
if len(sys.argv) != 2:
    print('Error, pass the number of cross-validation folds to the script')
    exit(-1)
		
num_folds = int(sys.argv[1])
if num_folds <= 0 or num_folds > 66:
    print('Error, the number of folds should be in the range [1,66]')
    exit(-1)

# Read PPG dataset	
records = read_datasets.getHUSMppg()	
num_recs = len(records)

# Length of the cross-validation folds
fold_len = num_recs // num_folds
leftovers = num_recs % num_folds
print(f'The size of each fold is {fold_len} records')

if leftovers > 0:
    print(f'There are {leftovers} unused records')
    print(f'This wont happen if {num_recs} is divisible by num_folds')

# Sampling frequency
Fs = 200

# Lists of parameters for GS
## To detect the systolic waves in PPG signals, the optimal solution was found to be
##     W1 = 111 ms, W2 = 667 ms and β = 2%.
# W1_list = [51, 57, 63, 69, 75, 81, 87, 93, 99, 105, 111]
# W2_list = [545, 560, 575, 590, 605, 620, 635, 650, 667, 680, 695]
# beta_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  
W1_list = [51, 111]
W2_list = [545, 695]
beta_list = [0, 0.1]

verbosity = True

# Generate num_folds splits of the loaded record and run grid search a number of times for each split
cv_parameters = []
cv_precisions = []
cv_recalls = []
for fold_i in range(num_folds):
    # Split the record
    fold_validation = records[fold_i * fold_len : (fold_i + 1) * fold_len]
    fold_train      = records[0 : fold_i * fold_len] + records[(fold_i + 1) * fold_len : len(records) - leftovers]
    
    # Search for TERMA parameters with grid search
    start_time = time.time()
    found_solution = optimization.grid_search_TERMA(train_records = fold_train,
                     W1_list = W1_list, W2_list = W2_list, beta_list = beta_list,
                     sampling_frequency = Fs, verbosity = verbosity)
    # if verbosity: print(f'Took {time.time() - start_time} sec')
    
    # Define TERMA model with the found parameters and compute validation metrics
    W1, W2, beta, _ = found_solution
    detector = TERMA_detector(W1, W2, beta)              
    tp, fp, fn = utilities.record_set_confusion_matrix(detector, fold_validation, Fs)
    
    if tp != 0:
        val_precision = tp / (tp + fp)
        val_recall    = tp / (tp + fn)
    else:
        val_precision = 0
        val_recall = 0
    
    # Store parameters found by GS and metrics for the validation fold
    cv_parameters.append( list(found_solution) )
    cv_precisions.append(val_precision)
    cv_recalls.append(val_recall)
    
if verbosity:
    print(f'Parameters { {np.size(cv_parameters) }}')
    print(cv_parameters)
    print(f'Precisions { {np.size(cv_precisions) }}')
    print(cv_precisions)
    print(f'Recalls    { {np.size(cv_recalls) }}')
    print(cv_recalls)

# Save results in binary files
if num_folds == 22:
		base_filename = f'search_results/LOSOCV_GS_TERMA_{num_folds}folds_'
else:
		base_filename = f'search_results/CV_GS_TERMA_{num_folds}folds_'
		
# 
np.save(base_filename + 'parameters.npy', cv_parameters)
np.save(base_filename + 'precisions.npy', cv_precisions)
np.save(base_filename + 'recalls.npy',    cv_recalls)
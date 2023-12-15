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
# Third party
import numpy as np

# Load PPG dataset
Fs = 200
records = np.array(read_datasets.getHUSMppg())
LSOCV_parameters = np.load('./search_results/LOSOCV_RS_SRMAC_22folds_30runs_parameters.npy')

# Select all parameters found by random search during LSOCV
parameters = LSOCV_parameters[:,:,-1,:]
parameters = parameters.reshape(660, 4)

# Split records in terms of protocol phase analysis
balke_indices = np.arange(0, 66, 3)
recovery_indices = np.arange(1, 66, 3)
rest_indices = np.arange(2, 66, 3)

balke_records = records[balke_indices]
recovery_records = records[recovery_indices]
rest_records = records[rest_indices]

balke_precisions = []
balke_recalls = []
recovery_precisions = []
recovery_recalls = []
rest_precisions = []
rest_recalls = []

for index, parameter_set in enumerate(parameters):
		print(f'Evaluating parameter set no. {index}')
		alpha_cross, alpha_fast, alpha_slow, thr = parameter_set
		detector = SRMAC_detector(alpha_cross, alpha_fast, alpha_slow, thr)
		
		# Get metrics for balke phase
		tp, fp, fn = utilities.record_set_confusion_matrix(detector, balke_records, Fs)
		balke_Pp = tp / (tp + fp)
		balke_SE = tp / (tp + fn)
		
		# Get metrics for recovery phase
		tp, fp, fn = utilities.record_set_confusion_matrix(detector, recovery_records, Fs)
		recovery_Pp = tp / (tp + fp)
		recovery_SE = tp / (tp + fn)
		
		# Get metrics for rest phase
		tp, fp, fn = utilities.record_set_confusion_matrix(detector, rest_records, Fs)
		rest_Pp = tp / (tp + fp)
		rest_SE = tp / (tp + fn)
		
		# Keep metrics
		balke_precisions.append(balke_Pp)
		recovery_precisions.append(recovery_Pp)
		rest_precisions.append(rest_Pp)
		
		recovery_recalls.append(recovery_SE)
		balke_recalls.append(balke_SE)
		rest_recalls.append(rest_SE)

print('Shapes')
print(np.shape(balke_precisions))
print(np.shape(recovery_precisions))
print(np.shape(rest_precisions))
print(np.shape(balke_recalls))
print(np.shape(recovery_recalls))
print(np.shape(rest_recalls))

print('\nAverage metrics')
print(f'Balke precision = {np.mean(balke_precisions)} ({np.std(balke_precisions, ddof=1)})')
print(f'Balke recall = {np.mean(balke_recalls)} ({np.std(balke_recalls, ddof=1)})')
print(f'Recovery precision = {np.mean(recovery_precisions)} ({np.std(recovery_precisions, ddof=1)})')
print(f'Recovery recall = {np.mean(recovery_recalls)} ({np.std(recovery_recalls, ddof=1)})')
print(f'Rest precision = {np.mean(rest_precisions)} ({np.std(rest_precisions, ddof=1)})')
print(f'Rest recall = {np.mean(rest_recalls)} ({np.std(rest_recalls, ddof=1)})')

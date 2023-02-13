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

import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print('Please inform the number of iterations of iterest (IOI) and the index of the IOI to be plotted')
    exit(-1)

ioimax = int(sys.argv[1])
ioi2plot = int(sys.argv[2])

if ioi2plot >= ioimax or ioi2plot < -1:
	print('The desired iteration of interest (IOI) to plot must be in the range set by the number of IOI or \'-1\' for no plotting')
	exit(-1)

# Loads multidimensional arrays with precision and recall values from cross-validation
val_precisions = np.load('../search_results/LOSOCV_RS_crossover_22folds_30runs_precisions.npy')
val_recalls = np.load('../search_results/LOSOCV_RS_crossover_22folds_30runs_recalls.npy')

# Array dimensions must be (num_folds, num_runs, iterations_of_interest)
print('Expected dimensions: (22, 30, # of IOI)')
print(f'Actual dimensions (P): {np.shape(val_precisions)}')
print(f'Actual dimensions (R): {np.shape(val_recalls)}')

print('\nStats on validation data')

fold_avg_precisions2plot = None
fold_avg_recalls2plot = None

for ioi in range(0, ioimax):
		print('\nIteration of interest with index ' + str(ioi))
		
		# There are 30 runs of the results for the iterations of interest
		fold_avg_precisions = np.sum(val_precisions[:,:,ioi], axis=1) / 30
		fold_avg_recalls = np.sum(val_recalls[:,:,ioi], axis=1) / 30

		if ioi == ioi2plot:
				print(f'ioi = {ioi2plot}')
				fold_avg_precisions2plot = list(fold_avg_precisions)
				fold_avg_recalls2plot = list(fold_avg_recalls)

		# COPD patients statistics
		c_precision_avg = np.mean(fold_avg_precisions[0:11])
		c_precision_std = np.std(fold_avg_precisions[0:11], ddof=1)
		c_recall_avg = np.mean(fold_avg_recalls[0:11])
		c_recall_std = np.std(fold_avg_recalls[0:11], ddof=1)

		print(f'COPD precision: {c_precision_avg} ({c_precision_std})')
		print(f'COPD recall: {c_recall_avg} ({c_recall_std})')

		# Healthy subjects statistics
		h_precision_avg = np.mean(fold_avg_precisions[11:22])
		h_precision_std = np.std(fold_avg_precisions[11:22], ddof=1)
		h_recall_avg = np.mean(fold_avg_recalls[11:22])
		h_recall_std = np.std(fold_avg_recalls[11:22], ddof=1)

		print(f'Healthy precision: {h_precision_avg} ({h_precision_std})')
		print(f'Healthy recall: {h_recall_avg} ({h_recall_std})')
		
		print(f'Overall average: {(c_precision_avg + c_recall_avg + h_precision_avg + h_recall_avg)/4}')

if ioi2plot == -1:
	exit()

fig, axs = plt.subplots(2, figsize=(15, 8))
folds_ticks = range(1,23)
xtick_labels = ['C'+r'$_1$', 'C'+r'$_2$', 'C'+r'$_3$', 'C'+r'$_4$', 'C'+r'$_5$', 'C'+r'$_6$',
                'C'+r'$_7$', 'C'+r'$_8$', 'C'+r'$_9$', 'C'+r'$_{10}$', 'C'+r'$_{11}$',
                'H'+r'$_1$', 'H'+r'$_2$', 'H'+r'$_3$', 'H'+r'$_4$', 'H'+r'$_5$', 'H'+r'$_6$',
                'H'+r'$_7$', 'H'+r'$_8$', 'H'+r'$_9$', 'H'+r'$_{10}$', 'H'+r'$_{11}$']

axs[0].set_title('Average precisions per subject')
barlist = axs[0].bar(folds_ticks, fold_avg_precisions2plot)
[ bar.set_color('#DE8484') for bar in barlist[0:11] ]
[ bar.set_color('#92BD77') for bar in barlist[11:22] ]

axs[0].set_ylim([0.89, 1.019])
axs[0].set_xticks(folds_ticks)
axs[0].set_xticklabels(xtick_labels)
axs[0].tick_params(axis='both', which='major', labelsize=16)

axs[1].set_title('Average recalls per subject')
barlist = axs[1].bar(folds_ticks, fold_avg_recalls2plot)
[ bar.set_color('#DE8484') for bar in barlist[0:11] ]
[ bar.set_color('#92BD77') for bar in barlist[11:22] ]

axs[1].set_ylim([0.89, 1.019])
axs[1].set_xticks(folds_ticks)
axs[1].set_xticklabels(xtick_labels)
axs[1].tick_params(axis='both', which='major', labelsize=16)

# plt.show()
plt.savefig(f'per_subject_THIS_RS_{ioi2plot}IOI.png')
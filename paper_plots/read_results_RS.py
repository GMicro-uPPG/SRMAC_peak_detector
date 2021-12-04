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
# import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Please inform the index of the desired iteration of interest')
    exit(-1)
    
ioi = int(sys.argv[1])
    
# Carrega array multidimensional com precisions
val_precisions = np.load('../search_results/LOSOCV_RS_22folds_30runs_precisions.npy')
val_recalls = np.load('../search_results/LOSOCV_RS_22folds_30runs_recalls.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22, 30, IOI)')
print(f'Dimensões obtidas (P): {np.shape(val_precisions)}')
print(f'Dimensões obtidas (R): {np.shape(val_recalls)}')

# Há 30 repetições dos resultados para as interações de interesse
# Iterações de interesse são [50, 100, 150, ..., 1000]
# Reservar soluções da iteração 150
fold_avg_precisions = []
fold_avg_recalls    = []

for subj_p, subj_r in zip(val_precisions, val_recalls):
    avg_precisions_of_interest = np.sum(subj_p, axis=0) / 30
    avg_recalls_of_interest = np.sum(subj_r, axis=0) / 30
    
    fold_avg_precisions.append(avg_precisions_of_interest[ioi])
    fold_avg_recalls.append(avg_recalls_of_interest[ioi])


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



# fig, axs = plt.subplots(2)
# folds_ticks = range(1,23)
# xtick_labels = ['C'+r'$_1$', 'C'+r'$_2$', 'C'+r'$_3$', 'C'+r'$_4$', 'C'+r'$_5$', 'C'+r'$_6$',
                # 'C'+r'$_7$', 'C'+r'$_8$', 'C'+r'$_9$', 'C'+r'$_{10}$', 'C'+r'$_{11}$',
                # 'H'+r'$_1$', 'H'+r'$_2$', 'H'+r'$_3$', 'H'+r'$_4$', 'H'+r'$_5$', 'H'+r'$_6$',
                # 'H'+r'$_7$', 'H'+r'$_8$', 'H'+r'$_9$', 'H'+r'$_{10}$', 'H'+r'$_{11}$']

# axs[0].set_title('Average precisions per subject')
# barlist = axs[0].bar(folds_ticks, fold_avg_precisions)
# [ bar.set_color('#DE8484') for bar in barlist[0:11] ]
# [ bar.set_color('#92BD77') for bar in barlist[11:22] ]

# axs[0].set_ylim([0.89, 1.019])
# axs[0].set_xticks(folds_ticks)
# axs[0].set_xticklabels(xtick_labels)
# axs[0].tick_params(axis='both', which='major', labelsize=16)

# axs[1].set_title('Average recalls per subject')
# barlist = axs[1].bar(folds_ticks, fold_avg_recalls)
# [ bar.set_color('#DE8484') for bar in barlist[0:11] ]
# [ bar.set_color('#92BD77') for bar in barlist[11:22] ]

# axs[1].set_ylim([0.89, 1.019])
# axs[1].set_xticks(folds_ticks)
# axs[1].set_xticklabels(xtick_labels)
# axs[1].tick_params(axis='both', which='major', labelsize=16)

# plt.show()
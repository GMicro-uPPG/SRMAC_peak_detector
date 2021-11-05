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
import matplotlib.pyplot as plt

# Carrega array multidimensional com precisions
val_precisions = np.load('../search_results/TERMA_GS_LOSOCV_22folds_precisions.npy')
val_recalls = np.load('../search_results/TERMA_GS_LOSOCV_22folds_recalls.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22)')
print(f'Dimensões obtidas (P): {np.shape(val_precisions)}')
print(f'Dimensões obtidas (R): {np.shape(val_recalls)}')


# COPD patients statistics
c_precision_avg = np.mean(val_precisions[0:11])
c_precision_std = np.std(val_precisions[0:11], ddof=1)
c_recall_avg = np.mean(val_recalls[0:11])
c_recall_std = np.std(val_recalls[0:11], ddof=1)

print(f'COPD precision: {c_precision_avg} ({c_precision_std})')
print(f'COPD recall: {c_recall_avg} ({c_recall_std})')

# Healthy subjects statistics
h_precision_avg = np.mean(val_precisions[11:22])
h_precision_std = np.std(val_precisions[11:22], ddof=1)
h_recall_avg = np.mean(val_recalls[11:22])
h_recall_std = np.std(val_recalls[11:22], ddof=1)

print(f'Healthy precision: {h_precision_avg} ({h_precision_std})')
print(f'Healthy recall: {h_recall_avg} ({h_recall_std})')


fig, axs = plt.subplots(2)
folds_ticks = range(1,23)
xtick_labels = ['C'+r'$_1$', 'C'+r'$_2$', 'C'+r'$_3$', 'C'+r'$_4$', 'C'+r'$_5$', 'C'+r'$_6$',
                'C'+r'$_7$', 'C'+r'$_8$', 'C'+r'$_9$', 'C'+r'$_{10}$', 'C'+r'$_{11}$',
                'H'+r'$_1$', 'H'+r'$_2$', 'H'+r'$_3$', 'H'+r'$_4$', 'H'+r'$_5$', 'H'+r'$_6$',
                'H'+r'$_7$', 'H'+r'$_8$', 'H'+r'$_9$', 'H'+r'$_{10}$', 'H'+r'$_{11}$']

# axs[0].set_title('Average precisions per subject')
barlist = axs[0].bar(folds_ticks, val_precisions)
[ bar.set_color('#DE8484') for bar in barlist[0:11] ]
[ bar.set_color('#92BD77') for bar in barlist[11:22] ]

axs[0].set_ylim([0.89, 1.019])
axs[0].set_xticks(folds_ticks)
axs[0].set_xticklabels(xtick_labels)
axs[0].tick_params(axis='both', which='major', labelsize=16)

# axs[1].set_title('Average recalls per subject')
barlist = axs[1].bar(folds_ticks, val_recalls)
[ bar.set_color('#DE8484') for bar in barlist[0:11] ]
[ bar.set_color('#92BD77') for bar in barlist[11:22] ]

axs[1].set_ylim([0.89, 1.019])
axs[1].set_xticks(folds_ticks)
axs[1].set_xticklabels(xtick_labels)
axs[1].tick_params(axis='both', which='major', labelsize=16)

plt.show()
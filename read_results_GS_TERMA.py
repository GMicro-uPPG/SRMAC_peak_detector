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
val_precisions = np.load('./search_results/TERMA_GS_LOSOCV_22folds_precisions.npy')
val_recalls = np.load('./search_results/TERMA_GS_LOSOCV_22folds_recalls.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22)')
print(f'Dimensões obtidas (P): {np.shape(val_precisions)}')
print(f'Dimensões obtidas (R): {np.shape(val_recalls)}')

avg_precision = np.sum(val_precisions) / 22
avg_recall    = np.sum(val_recalls) / 22
avg_avg = (avg_precision + avg_recall)/2

print(f'Avg precision: {avg_precision}, avg recall: {avg_recall}, avg avg: {avg_avg}')


fig, axs = plt.subplots(2)
folds_ticks = range(1,23)

axs[0].set_title('Average precisions per subject')
axs[0].bar(folds_ticks, val_precisions)
axs[0].set_ylim([0.89, 1.03])
axs[0].set_xticks(folds_ticks)

axs[1].set_title('Average recalls per subject')
axs[1].bar(folds_ticks, val_recalls)
axs[1].set_ylim([0.89, 1.03])
axs[1].set_xticks(folds_ticks)

plt.show()
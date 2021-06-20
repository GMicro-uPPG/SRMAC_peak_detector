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
val_precisions = np.load('LOSOCV_22folds_30runs_precisions.npy')
val_recalls = np.load('LOSOCV_22folds_30runs_recalls.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22, 30, 20)')
print(f'Dimensões obtidas (P): {np.shape(val_precisions)}')
print(f'Dimensões obtidas (R): {np.shape(val_recalls)}')

# Há 30 repetições dos resultados para as interações de interesse
# Iterações de interesse são [50, 100, 150, ..., 1000]
all_avg_precisions = []
all_avg_recalls    = []

for subj_p, subj_r in zip(val_precisions, val_recalls):
    avg_precisions_of_interest = np.sum(subj_p, axis=0) / 30
    avg_recalls_of_interest = np.sum(subj_r, axis=0) / 30
    
    all_avg_precisions.append(list(avg_precisions_of_interest))
    all_avg_recalls.append(list(avg_recalls_of_interest))

overall_avg_precisions = np.sum(all_avg_precisions, axis=0) / 22
overall_avg_recalls    = np.sum(all_avg_recalls, axis = 0) / 22

iterations_of_interest = [i*50 for i in range(1,21)]
iterations_ticks = range(len(iterations_of_interest))

fig, axs = plt.subplots(2)

axs[0].set_title('Average precisions')
axs[0].bar(iterations_ticks, overall_avg_precisions)
axs[0].set_ylim([0.98, 0.985])
axs[0].set_xticks(iterations_ticks)
axs[0].set_xticklabels([str(x) for x in iterations_of_interest])

axs[1].set_title('Average recalls')
axs[1].bar(iterations_ticks, overall_avg_recalls)
axs[1].set_ylim([0.98, 0.985])
axs[1].set_xticks(iterations_ticks)
axs[1].set_xticklabels([str(x) for x in iterations_of_interest])

plt.show()
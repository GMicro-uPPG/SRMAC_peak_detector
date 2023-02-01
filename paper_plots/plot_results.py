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

# Loads arrays with precision and recall values from cross-validation
## Validation results for the proposed model
val_precisions_this = np.load('../search_results/LOSOCV_RS_22folds_30runs_precisions.npy')
val_recalls_this = np.load('../search_results/LOSOCV_RS_22folds_30runs_recalls.npy')

## Validation results for TERMA
val_precisions_terma = np.load('../search_results/TERMA_GS_LOSOCV_22folds_precisions.npy')
val_recalls_terma = np.load('../search_results/TERMA_GS_LOSOCV_22folds_recalls.npy')

# Dimensionality checking
print('Expected dimensions for our results: (22, 30, # of IOI)')
print(f'Actual dimensions (P): {np.shape(val_precisions_this)}')
print(f'Actual dimensions (R): {np.shape(val_recalls_this)}')

print('\nExpected dimensions for TERMA\'s results: (22)')
print(f'Actual dimensions (P): {np.shape(val_precisions_terma)}')
print(f'Actual dimensions (R): {np.shape(val_recalls_terma)}')

# Split data, considering origin method, health condition and metric
## TERMA 
pre_copd_terma = val_precisions_terma[0:11]
pre_healthy_terma = val_precisions_terma[11:22]
rec_copd_terma = val_recalls_terma[0:11]
rec_healthy_terma = val_recalls_terma[11:22]

## THIS
pre_copd_this = np.sum(val_precisions_this[0:11,:,-1], axis=1) / 30
pre_healthy_this = np.sum(val_precisions_this[11:22,:,-1], axis=1) / 30
rec_copd_this = np.sum(val_recalls_this[0:11,:,-1], axis=1) / 30
rec_healthy_this = np.sum(val_recalls_this[11:22,:,-1], axis=1) / 30

# Summary statistics to check for correctness
# print('\nSTATS FOR TERMA')
print(f'COPD precision: {np.mean(pre_copd_terma)} {np.std(pre_copd_terma, ddof=1)}')
print(f'COPD recall: {np.mean(rec_copd_terma)} {np.std(rec_copd_terma, ddof=1)}')
print(f'Healthy precision: {np.mean(pre_healthy_terma)} {np.std(pre_healthy_terma, ddof=1)}')
print(f'Healthy recall: {np.mean(rec_healthy_terma)} {np.std(rec_healthy_terma, ddof=1)}')

print('\nSTATS FOR THIS')
print(f'COPD precision: {np.mean(pre_copd_this)} {np.std(pre_copd_this, ddof=1)}')
print(f'COPD recall: {np.mean(rec_copd_this)} {np.std(rec_copd_this, ddof=1)}')
print(f'Healthy precision: {np.mean(pre_healthy_this)} {np.std(pre_healthy_this, ddof=1)}')
print(f'Healthy recall: {np.mean(rec_healthy_this)} {np.std(rec_healthy_this, ddof=1)}')

# There are 4 plots, from the cartesian product (precision, recall)x(healthy, COPD)
bar_width = 0.35
subj_ticks = np.arange(1,12)
healthy_labels = ['H'+r'$_1$', 'H'+r'$_2$', 'H'+r'$_3$', 'H'+r'$_4$', 'H'+r'$_5$', 'H'+r'$_6$',
                'H'+r'$_7$', 'H'+r'$_8$', 'H'+r'$_9$', 'H'+r'$_{10}$', 'H'+r'$_{11}$']
COPD_labels = ['C'+r'$_1$', 'C'+r'$_2$', 'C'+r'$_3$', 'C'+r'$_4$', 'C'+r'$_5$', 'C'+r'$_6$',
                'C'+r'$_7$', 'C'+r'$_8$', 'C'+r'$_9$', 'C'+r'$_{10}$', 'C'+r'$_{11}$']

## (Precision, healthy)
fig, axs = plt.subplots(1, figsize=(15, 8))
barlist_this = axs.bar(subj_ticks, pre_healthy_this, width = bar_width, label='This')
barlist_terma = axs.bar(subj_ticks + bar_width, pre_healthy_terma, width = bar_width, label='TERMA')
[ bar.set_color('#DE8484') for bar in barlist_this ]
[ bar.set_color('#92BD77') for bar in barlist_terma ]

axs.set_title('Precisions for healthy subjects')
axs.set_ylim([0.94, 1.001])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(healthy_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig(f'')	
plt.show()

## (Precision, COPD)
fig, axs = plt.subplots(1, figsize=(15, 8))
barlist_this = axs.bar(subj_ticks, pre_copd_this, width = bar_width, label='This')
barlist_terma = axs.bar(subj_ticks + bar_width, pre_copd_terma, width = bar_width, label='TERMA')
[ bar.set_color('#DE8484') for bar in barlist_this ]
[ bar.set_color('#92BD77') for bar in barlist_terma ]

axs.set_title('Precisions for COPD patients')
axs.set_ylim([0.94, 1.001])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(COPD_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig(f'')	
plt.show()

## (Recall, healthy)
fig, axs = plt.subplots(1, figsize=(15, 8))
barlist_this = axs.bar(subj_ticks, rec_healthy_this, width = bar_width, label='This')
barlist_terma = axs.bar(subj_ticks + bar_width, rec_healthy_terma, width = bar_width, label='TERMA')
[ bar.set_color('#DE8484') for bar in barlist_this ]
[ bar.set_color('#92BD77') for bar in barlist_terma ]

axs.set_title('Recalls for healthy subjects')
axs.set_ylim([0.94, 1.001])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(healthy_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig(f'')	
plt.show()

## (Recall, COPD)
fig, axs = plt.subplots(1, figsize=(15, 8))
barlist_this = axs.bar(subj_ticks, rec_copd_this, width = bar_width, label='This')
barlist_terma = axs.bar(subj_ticks + bar_width, rec_copd_terma, width = bar_width, label='TERMA')
[ bar.set_color('#DE8484') for bar in barlist_this ]
[ bar.set_color('#92BD77') for bar in barlist_terma ]

axs.set_title('Recalls for COPD patients')
axs.set_ylim([0.94, 1.001])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(COPD_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig(f'')	
plt.show()
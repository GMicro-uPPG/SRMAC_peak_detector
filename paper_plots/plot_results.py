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

import numpy as np
import matplotlib.pyplot as plt

# Loads arrays with precision and recall values from cross-validation
## Validation results for the proposed model
precisions_SRMAC = np.load('../search_results/LOSOCV_RS_SRMAC_22folds_30runs_precisions.npy')
recalls_SRMAC = np.load('../search_results/LOSOCV_RS_SRMAC_22folds_30runs_recalls.npy')

## Validation results for TERMA
precisions_terma = np.load('../search_results/LOSOCV_GS_TERMA_22folds_precisions.npy')
recalls_terma = np.load('../search_results/LOSOCV_GS_TERMA_22folds_recalls.npy')

# Dimensionality checking
print('Expected dimensions for our results: (22, 30, # of IOI)')
print(f'Actual dimensions (P): {np.shape(precisions_SRMAC)}')
print(f'Actual dimensions (R): {np.shape(recalls_SRMAC)}')

print('\nExpected dimensions for TERMA\'s results: (22)')
print(f'Actual dimensions (P): {np.shape(precisions_terma)}')
print(f'Actual dimensions (R): {np.shape(recalls_terma)}')

# Split data, considering origin method, health condition and metric
## TERMA 
pre_copd_terma = precisions_terma[0:11]
pre_healthy_terma = precisions_terma[11:22]
rec_copd_terma = recalls_terma[0:11]
rec_healthy_terma = recalls_terma[11:22]

## THIS
pre_copd_SRMAC = np.sum(precisions_SRMAC[0:11,:,-1], axis=1) / 30
pre_healthy_SRMAC = np.sum(precisions_SRMAC[11:22,:,-1], axis=1) / 30
rec_copd_SRMAC = np.sum(recalls_SRMAC[0:11,:,-1], axis=1) / 30
rec_healthy_SRMAC = np.sum(recalls_SRMAC[11:22,:,-1], axis=1) / 30

# Summary statistics to check for correctness
print('\nSTATS FOR TERMA')
print(f'COPD precision: {np.mean(pre_copd_terma)} {np.std(pre_copd_terma, ddof=1)}')
print(f'COPD recall: {np.mean(rec_copd_terma)} {np.std(rec_copd_terma, ddof=1)}')
print(f'Healthy precision: {np.mean(pre_healthy_terma)} {np.std(pre_healthy_terma, ddof=1)}')
print(f'Healthy recall: {np.mean(rec_healthy_terma)} {np.std(rec_healthy_terma, ddof=1)}')
print(f'Overall accuracy: {(np.mean(pre_copd_terma) + np.mean(rec_copd_terma) + np.mean(pre_healthy_terma) + np.mean(rec_healthy_terma))/4}')

print('\nSTATS FOR SRMAC')
print(f'COPD precision: {np.mean(pre_copd_SRMAC)} {np.std(pre_copd_SRMAC, ddof=1)}')
print(f'COPD recall: {np.mean(rec_copd_SRMAC)} {np.std(rec_copd_SRMAC, ddof=1)}')
print(f'Healthy precision: {np.mean(pre_healthy_SRMAC)} {np.std(pre_healthy_SRMAC, ddof=1)}')
print(f'Healthy recall: {np.mean(rec_healthy_SRMAC)} {np.std(rec_healthy_SRMAC, ddof=1)}')
print(f'Overall accuracy: {(np.mean(pre_copd_SRMAC) + np.mean(rec_copd_SRMAC) + np.mean(pre_healthy_SRMAC) + np.mean(rec_healthy_SRMAC))/4}')

# There are 4 plots, from the cartesian product (precision, recall)x(healthy, COPD)
bar_width = 0.4
bar_offset = bar_width/2 + 0.02
y_max = 1.01
y_min = 0.9
color_SRMAC = 'mediumseagreen'
color_TERMA = 'slateblue'
figure_proportion = (15, 5)

subj_ticks = np.arange(1,12)
healthy_labels = ['H'+r'$_1$', 'H'+r'$_2$', 'H'+r'$_3$', 'H'+r'$_4$', 'H'+r'$_5$', 'H'+r'$_6$',
                  'H'+r'$_7$', 'H'+r'$_8$', 'H'+r'$_9$', 'H'+r'$_{10}$', 'H'+r'$_{11}$']
COPD_labels = ['C'+r'$_1$', 'C'+r'$_2$', 'C'+r'$_3$', 'C'+r'$_4$', 'C'+r'$_5$', 'C'+r'$_6$',
               'C'+r'$_7$', 'C'+r'$_8$', 'C'+r'$_9$', 'C'+r'$_{10}$', 'C'+r'$_{11}$']


## (Precision, healthy)
fig, axs = plt.subplots(1, figsize = figure_proportion)
barlist_SRMAC = axs.bar(subj_ticks - bar_offset, pre_healthy_SRMAC, width = bar_width, label='SRMAC')
barlist_terma = axs.bar(subj_ticks + bar_offset, pre_healthy_terma, width = bar_width, label='TERMA')
[ bar.set_color(color_SRMAC) for bar in barlist_SRMAC ]
[ bar.set_color(color_TERMA) for bar in barlist_terma ]
axs.set_title('Precisions for healthy subjects', fontsize=18)
axs.set_ylim([y_min, y_max])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(healthy_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(f'precisions_healthy_subs.png', bbox_inches='tight')	

## (Precision, COPD)
fig, axs = plt.subplots(1, figsize = figure_proportion)
barlist_SRMAC = axs.bar(subj_ticks - bar_offset, pre_copd_SRMAC, width = bar_width, label='SRMAC')
barlist_terma = axs.bar(subj_ticks + bar_offset, pre_copd_terma, width = bar_width, label='TERMA')
[ bar.set_color(color_SRMAC) for bar in barlist_SRMAC ]
[ bar.set_color(color_TERMA) for bar in barlist_terma ]
axs.set_title('Precisions for COPD patients', fontsize=18)
axs.set_ylim([y_min, y_max])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(COPD_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(f'precisions_copd_subs.png', bbox_inches='tight')	

## (Recall, healthy)
fig, axs = plt.subplots(1, figsize = figure_proportion)
barlist_SRMAC = axs.bar(subj_ticks - bar_offset, rec_healthy_SRMAC, width = bar_width, label='SRMAC')
barlist_terma = axs.bar(subj_ticks + bar_offset, rec_healthy_terma, width = bar_width, label='TERMA')
[ bar.set_color(color_SRMAC) for bar in barlist_SRMAC ]
[ bar.set_color(color_TERMA) for bar in barlist_terma ]
axs.set_title('Recalls for healthy subjects', fontsize=18)
axs.set_ylim([y_min, y_max])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(healthy_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(f'recalls_healthy_subs.png', bbox_inches='tight')

## (Recall, COPD)
fig, axs = plt.subplots(1, figsize = figure_proportion)
barlist_SRMAC = axs.bar(subj_ticks - bar_offset, rec_copd_SRMAC, width = bar_width, label='SRMAC')
barlist_terma = axs.bar(subj_ticks + bar_offset, rec_copd_terma, width = bar_width, label='TERMA')
[ bar.set_color(color_SRMAC) for bar in barlist_SRMAC ]
[ bar.set_color(color_TERMA) for bar in barlist_terma ]
axs.set_title('Recalls for COPD patients', fontsize=18)
axs.set_ylim([y_min, y_max])
axs.set_xticks(subj_ticks)
axs.set_xticklabels(COPD_labels)
axs.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(f'recalls_copd_subs.png', bbox_inches='tight')	

# plt.show()
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

if len(sys.argv) != 2:
    print("Erro, informe o sujeito")
    exit(-1)
    
subj = int(sys.argv[1])
if subj >= 22:
    print("Erro, sujeito nao existe")
    exit(-1)

# Carrega array multidimensional com precisions
val_precisions = np.load('LOSOCV_22folds_30runs_precisions.npy')
val_recalls = np.load('LOSOCV_22folds_30runs_recalls.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22, 30, 20)')
print(f'Dimensões obtidas (P): {np.shape(val_precisions)}')
print(f'Dimensões obtidas (R): {np.shape(val_recalls)}')

# Um fold representa umm sujeito
subj_p = val_precisions[subj]
subj_r = val_recalls[subj]
print(f'Dimensões de um fold: P = {np.shape(subj_p)}, R = {np.shape(subj_r)}')

# Há 30 repetições dos resultados para as interações de interesse
# Iterações de interesse são [50, 100, 150, ..., 1000]
# Obtenção da precisão média entre repretições para as iterações de interesse (considerando o sujeito 0):
avg_precisions_of_interest = np.sum(subj_p, axis=0) / 30
avg_recalls_of_interest = np.sum(subj_r, axis=0) / 30
print()
print('Iterações de interesse')
print([i*50 for i in range(1,21)])
print(f'\nPrecisões médias para o sujeito {subj}')
print(f'Dimensões: {np.shape(avg_precisions_of_interest)}')
print(avg_precisions_of_interest)
print(f'\nRecalls médias para o sujeito {subj}')
print(f'Dimensões: {np.shape(avg_recalls_of_interest)}')
print(avg_recalls_of_interest)

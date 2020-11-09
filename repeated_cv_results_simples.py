import numpy as np

# Carrega array multidimensional com precisions
val_precisions = np.load('LOSOCV_22folds_30runs_precisions.npy')

# Dimensões do array devem ser (num_folds, num_runs, iterations_of_interest)
print('Dimensões esperadas: (22, 30, 20)')
print(f'Dimensões obtidas: {np.shape(val_precisions)}')

# Um fold representa umm sujeito
subj = 0
fold_subj = val_precisions[subj]
print(f'Dimensões de um fold: {np.shape(fold_subj)}')

# Há 30 repetições dos resultados para as interações de interesse
# Iterações de interesse são [50, 100, 150, ..., 1000]
# Obtenção da precisão média entre repretições para as iterações de interesse (considerando o sujeito 0):
avg_precisions_of_interest = np.sum(fold_subj, axis=0) / 30
print()
print('Iterações de interesse')
print([i*50 for i in range(1,21)])
print()
print(f'Precisão média para o sujeito {subj}')
print(f'Dimensões: {np.shape(avg_precisions_of_interest)}')
print(avg_precisions_of_interest)
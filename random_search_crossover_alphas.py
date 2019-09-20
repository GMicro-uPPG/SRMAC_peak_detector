#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas using regularized confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
#from plot import *

# Record example
# name = records[0].name              # Record name: string
# ppg = records[0].ppg                # Record ppg: [x_ppg, ppg]
# hrv = records[0].hrv                # Record hrv: [x_hrv, hrv]
# plotPPG(name, ppg, hrv)             # Plot ppg signal and peak points

# Use 30 records to train model
train_records = records[0:30]

# Random search of alphas, using regularized confusion matrix-based cost
peak_detector = crossover_detector()
# Parameters
C = 3                                          # Regularization hyperparameter
num_iterations = 100                          # Number of random search iterations
# Optimization
solution_archive = np.zeros((num_iterations,3))
for iteration in range(num_iterations):
    print("[Search iteration ", iteration, "]")
    # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha > slow alpha
    alpha_fast = np.random.uniform(0, 1)
    alpha_slow = np.random.uniform(alpha_fast, 1)   
    peak_detector.set_parameters(alpha_fast, alpha_slow)
    cost = peak_detector.total_regularized_cost(train_records, C)
    # Keep solutions in a matrix
    solution_archive[iteration, :] = [alpha_fast, alpha_slow, cost]

# Sort solutions according to the costs
solution_archive = solution_archive[solution_archive[:,-1].argsort()]
best_solution = solution_archive[0]
#print(solution_archive)
pkl.dump(solution_archive, open("solution_archive.data","wb"))
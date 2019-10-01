#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas using regularized confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time
#from plot import *


try:
    # Record example
    # name = records[0].name              # Record name: string
    # ppg = records[0].ppg                # Record ppg: [x_ppg, ppg]
    # hrv = records[0].hrv                # Record hrv: [x_hrv, hrv]
    # plotPPG(name, ppg, hrv)             # Plot ppg signal and peak points

    # Use 30 records to train model
    train_records = records[0:5]
    print('\nrecords[0:5]')

    # Random search of alphas, using regularized confusion matrix-based cost
    peak_detector = crossover_detector()
    # Parameters
    C = 1                                          # Regularization hyperparameter
    print('\nC = ' + str(C))

    num_iterations = 6000 # 10000                         # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))

    # Optimization
    #solution_archive = np.zeros((num_iterations,3))
    best_solution = []
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha < slow alpha
        alpha_fast = np.random.uniform(0, 1)
        alpha_slow = np.random.uniform(alpha_fast, 1)   
        peak_detector.set_parameters(alpha_fast, alpha_slow)
        cost = peak_detector.total_regularized_cost(train_records, C)
        print('[randomized] alpha_fast: ', peak_detector.alpha_fast, ', alpha_slow: ', peak_detector.alpha_slow, ', cost: ', cost)
        
        # Keep solutions in a matrix
        if iteration == 0:
            best_solution = [alpha_fast, alpha_slow, cost]
        elif cost < best_solution[-1]:
            best_solution = [alpha_fast, alpha_slow, cost]
        print('[current best solution] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], ', cost: ', best_solution[-1])
        #solution_archive[iteration, :] = [alpha_fast, alpha_slow, cost]

    # Sort solutions according to the costs
    #solution_archive = solution_archive[solution_archive[:,-1].argsort()]
    #best_solution = solution_archive[0]
    #print(solution_archive)
    #pkl.dump(solution_archive, open("solution_archive.data","wb"))


    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
#/try


except IOError:
    print('Error: An error occurred trying to read the file.\n')
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
except ValueError:
    print('Error: Non-numeric data found in the file.\n')
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
except ImportError:
    print('Error: No module found.\n')
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
except EOFError:
    print('Error: Why did you do an EOF on me?\n')
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
except KeyboardInterrupt:
    print('Error: You cancelled the operation.\n')
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
except Exception as e:
    print('An error occurred:', e)
    print('\nLast timestamp: ' + str(time.getTimestamp()))
    print('Last time: ' + str(time.getTime()))
#/except
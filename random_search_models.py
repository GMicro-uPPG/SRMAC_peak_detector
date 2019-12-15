#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and derivtive threshold using confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time
from plot import *

try:
    # Record example
    #rec = 21
    #name = records[rec].name              # Record name: string
    #ppg = records[rec].ppg                # Record ppg: [x_ppg, ppg]
    #beats = records[rec].beats            # Record beats: [x_beats, beats]
    #plotPPG(name, ppg, beats)             # Plot ppg signal and peak points


    # Use 30 records to train model and 30 to test it
    train_records = records[0:30]
    test_records = records[30:60]
    
    print('\nrecords[0:30]')

    # Random search of alphas, using regularized confusion matrix-based cost
    peak_detector = crossover_detector()
    # Parameters
    C = 0                                         # Regularization hyperparameter
    print('\nC = ' + str(C))

    num_iterations = 1000                            # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))
    
    # Optimization
    #solution_archive = np.zeros((num_iterations,3))
    best_solution = [0, 0, float('inf')]
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover
        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha < slow alpha
        alpha_fast = np.random.uniform(0.9, 1)
        alpha_slow = np.random.uniform(alpha_fast, 1)
        peak_detector.set_parameters_cross(alpha_fast, alpha_slow)
        cost = peak_detector.total_regularized_cost(train_records, C, "crossover")
        print('[randomized] alpha_fast: ', alpha_fast, ', alpha_slow: ', alpha_slow,', cost: ', cost)
        # Keep solutions in a matrix
        if cost < best_solution[-1]:
            best_solution = [alpha_fast, alpha_slow, cost]
        print('[current best solution] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], ', cost: ', best_solution[-1])
        
        ## Optimize variance
        # Randomize parameters
        # var_alpha = np.random.uniform(0,1)
        # var_threshold = np.random.uniform(0,300)
        # peak_detector.set_parameters_var(var_alpha, var_threshold)
        # cost = peak_detector.total_regularized_cost(train_records, C, 'variance')
        # print('[randomized] var_alpha: ', var_alpha, ', threshold: ', var_threshold, ', cost: ', cost)
        # if cost < best_solution[-1]:
            # best_solution = [var_alpha, var_threshold, cost]
        # print('[current best] var_alpha: ', best_solution[0], 'threshold: ', best_solution[1], ', cost: ', best_solution[-1])
        
        ## Optimize mixed
        # alpha_fast = np.random.uniform(0.9, 1)
        # alpha_slow = np.random.uniform(alpha_fast, 1)
        # var_alpha = np.random.uniform(0,1)
        # avg_alpha = np.random.uniform(0,1)
        # var_threshold = np.random.uniform(0,100)
        # peak_detector.set_parameters_mix(alpha_fast, alpha_slow, var_alpha, avg_alpha, var_threshold)
        # cost = peak_detector.total_regularized_cost(train_records, C, 'mix')
        # print('[randomized] alpha_fast: ', alpha_fast, ', alpha_slow: ', alpha_slow, 'var_alpha: ', var_alpha, ', avg_alpha: ', avg_alpha, 'threshold: ', var_threshold, ', cost: ', cost)
        # if cost < best_solution[-1]:
            # best_solution = [alpha_fast, alpha_slow, var_alpha, avg_alpha, var_threshold, cost]
        # print('[current best] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], 'var_alpha: ', best_solution[2], ', avg_alpha: ', best_solution[3], 'threshold: ', best_solution[4], ', cost: ', best_solution[-1])
        
    
    peak_detector.set_parameters_cross(best_solution[0], best_solution[1])
    # peak_detector.set_parameters_var(best_solution[0], best_solution[1], best_solution[2])
    # peak_detector.set_parameters_mix(best_solution[0], best_solution[1], best_solution[2], best_solution[3], best_solution[4])
    
    train_confusion_matrix = peak_detector.record_set_confusion_matrix(train_records)
    test_confusion_matrix = peak_detector.record_set_confusion_matrix(test_records)
    
    print('Train set confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
    print('Test set confusion matrix: [TP,TN,FP,FN]' + str(test_confusion_matrix))
    

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
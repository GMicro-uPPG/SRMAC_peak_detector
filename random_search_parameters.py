#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and derivtive threshold using confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from read_ppg_mimic import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time
from plot import *


try:
    # Record example
    #rec = 21
    #name = records[rec].name              # Record name: string
    #ppg = records[rec].ppg                # Record ppg: [x_ppg, ppg]
    #hrv = records[rec].hrv                # Record hrv: [x_hrv, hrv]
    #plotPPG(name, ppg, hrv)             # Plot ppg signal and peak points


    # Use 30 records to train model and 30 to test it
    train_records = records[0:30]
    test_records = records[30:60]
    
    print('\nrecords[0:30]')

    # Random search of alphas, using regularized confusion matrix-based cost
    peak_detector = crossover_detector()
    # Parameters
    C = 3.0                                         # Regularization hyperparameter
    diff_max_threshold = 2.0                        # Maximum difference threshold value
    print('\nC = ' + str(C))

    num_iterations = 300                            # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))

    
    # Optimization
    #solution_archive = np.zeros((num_iterations,3))
    best_solution = []
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha < slow alpha
        alpha_fast = np.random.uniform(0, 1)
        alpha_slow = np.random.uniform(alpha_fast, 1)   
        difference_threshold = np.random.uniform(0.0, diff_max_threshold)
        peak_detector.set_parameters(alpha_fast, alpha_slow, difference_threshold)
        cost = peak_detector.total_regularized_cost(train_records, C)
        print('[randomized] alpha_fast: ', peak_detector.alpha_fast, ', alpha_slow: ', peak_detector.alpha_slow, ', difference threshold: ', peak_detector.difference_threshold, ', cost: ', cost)
        
        # Keep solutions in a matrix
        if iteration == 0:
            best_solution = [alpha_fast, alpha_slow, cost]
        elif cost < best_solution[-1]:
            best_solution = [alpha_fast, alpha_slow, cost]
        print('[current best solution] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], ', difference threshold: ', best_solution[2], ', cost: ', best_solution[-1])
        #solution_archive[iteration, :] = [alpha_fast, alpha_slow, cost]

    # Sort solutions according to the costs
    #solution_archive = solution_archive[solution_archive[:,-1].argsort()]
    #best_solution = solution_archive[0]
    #print(solution_archive)
    #pkl.dump(solution_archive, open("solution_archive.data","wb"))
    
    peak_detector.set_parameters(best_solution[0], best_solution[1], best_solution[2])
    train_confusion_matrix = peak_detector.record_confusion_matrix(train_records)
    test_confusion_matrix = peak_detector.record_confusion_matrix(test_records)
    print('Train set confusion matrix: [TP,TN,FP,FN]' + train_confusion_matrix)
    print('Test set confusion matrix: [TP,TN,FP,FN]' + test_confusion_matrix)
    

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
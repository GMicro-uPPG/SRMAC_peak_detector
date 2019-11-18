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
    
    
    # Bagging solution Archive
    # solution_archive[0] <=> best bagging_alpha_fasts
    # solution_archive[1] <=> best bagging_alpha_slows
    # solution_archive[2] <=> best bagging_costs
    num_members = 10
    solution_archive = np.ones((num_members, 3))

    # Optimization
    #solution_archive = np.zeros((num_iterations,3))
    best_solution = []
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover
        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha < slow alpha
        # alpha_fast = np.random.uniform(0, 1)
        # alpha_slow = np.random.uniform(alpha_fast, 1)
        # peak_detector.set_parameters_cross(alpha_fast, alpha_slow)
        # cost = peak_detector.total_regularized_cost(train_records, C, "crossover")
        # print('[randomized] alpha_fast: ', alpha_fast, ', alpha_slow: ', alpha_slow,', cost: ', cost)
        # # Keep solutions in a matrix
        # if iteration == 0:
            # best_solution = [alpha_fast, alpha_slow, cost]
        # elif cost < best_solution[-1]:
            # best_solution = [alpha_fast, alpha_slow, cost]
        # print('[current best solution] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], ', cost: ', best_solution[-1])
        # #solution_archive[iteration, :] = [alpha_fast, alpha_slow, cost]
        
        
        # ## Optimize crossover ensembles
        # # Keep num_members best solutions and build voting ensemble by bootstrap sampling (bagging)
        
        # bagging_alpha_fasts = np.random.uniform(0, 1, num_members)
        # bagging_alpha_slows = np.random.uniform(bagging_alpha_fasts, num_members*[1], num_members)
        # local_archive = []
        # for i in range(0, num_members):
            # peak_detector.set_parameters_cross(bagging_alpha_fasts[i], bagging_alpha_slows[i])
            # # Resamples train set with repick to generate diverse models
            # bootstrap_indices = np.random.randint(0, len(train_records), len(train_records))
            # train_bootstrap_records = np.array(train_records)[bootstrap_indices]
            # cost = peak_detector.total_regularized_cost(train_bootstrap_records, C, "crossover")
            # local_archive.append([bagging_alpha_fasts[i], bagging_alpha_slows[i], cost])
            
        
        # # Merge, sort and remove worst
        # solution_archive = np.append(solution_archive, local_archive, axis = 0)                     # Append new solutions to the Archive        
        # solution_archive = solution_archive[solution_archive[:, -1].argsort()]                      # Sort solution archive according to the fitness of each solution
        # solution_archive = solution_archive[0:num_members, :]                                       # Remove worst solutions           
        
        # print('[Current archive]')
        # print(solution_archive)
        
        ## Optimize variance
        # Randomize parameters
        # var_alpha = np.random.uniform(0,1)
        # var_threshold = np.random.uniform(0,300)
        # peak_detector.set_parameters_var(var_alpha, var_threshold)
        # cost = peak_detector.total_regularized_cost(train_records, C, 'variance')
        # print('[randomized] var_alpha: ', var_alpha, ', threshold: ', var_threshold, ', cost: ', cost)
        # if iteration == 0:
            # best_solution = [var_alpha, var_threshold, cost]
        # elif cost < best_solution[-1]:
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
        # if iteration == 0:
            # best_solution = [alpha_fast, alpha_slow, var_alpha, avg_alpha, var_threshold, cost]
        # elif cost < best_solution[-1]:
            # best_solution = [alpha_fast, alpha_slow, var_alpha, avg_alpha, var_threshold, cost]
        # print('[current best] alpha_fast: ', best_solution[0], ', alpha_slow: ', best_solution[1], 'var_alpha: ', best_solution[2], ', avg_alpha: ', best_solution[3], 'threshold: ', best_solution[4], ', cost: ', best_solution[-1])
        
    
    # peak_detector.set_parameters_cross(best_solution[0], best_solution[1])
    # peak_detector.set_parameters_var(best_solution[0], best_solution[1], best_solution[2])
    # peak_detector.set_parameters_mix(best_solution[0], best_solution[1], best_solution[2], best_solution[3], best_solution[4])
    
    # train_confusion_matrix = peak_detector.record_set_confusion_matrix(train_records)
    # test_confusion_matrix = peak_detector.record_set_confusion_matrix(test_records)
    
    # pkl.dump(solution_archive, open("solution_archive.data","wb"))
    # train_confusion_matrix = peak_detector.bagging_records_confusion_matrix(solution_archive, train_records)
    # test_confusion_matrix = peak_detector.bagging_records_confusion_matrix(solution_archive, test_records)
    
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
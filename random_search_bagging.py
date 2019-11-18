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
    C = 10                                         # Regularization hyperparameter
    print('\nC = ' + str(C))
    num_iterations = 1000                            # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))
    
    # Bagging solution Archive
    # solution_archive[0] <=> best bagging_alpha_fasts
    # solution_archive[1] <=> best bagging_alpha_slows
    # solution_archive[2] <=> best bagging_costs
    ensemble_size = 10
    
    solution_archive = np.ones((ensemble_size, 3))
    
    # Generate boostrap fixed indices (ensemble_size x input length matrix)
    bootstrap_indices = np.random.randint(low = 0, high=len(train_records), size=(ensemble_size, len(train_records)))
    
    solution_archive = np.zeros((ensemble_size,3))
    solution_archive[:, -1] = float('inf')
    
    best_solution = [0, 0, ]
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover ensembles
        # Keep ensemble_size best solutions and build voting ensemble by bootstrap sampling (bagging)
        bagging_alpha_fasts = np.random.uniform(0, 1, ensemble_size)
        bagging_alpha_slows = np.random.uniform(bagging_alpha_fasts, ensemble_size*[1], ensemble_size)
        local_archive = []
        for i in range(0, ensemble_size):
            # print("Ensemble member " + str(i))
            peak_detector.set_parameters_cross(bagging_alpha_fasts[i], bagging_alpha_slows[i])
            # Resamples train set with repick to generate diverse models
            bagging_indices = bootstrap_indices[i]
            # print("bagging indices:" + str(bagging_indices))
            train_bootstrap_records = np.array(train_records)[bagging_indices]
            cost = peak_detector.total_regularized_cost(train_bootstrap_records, C, "crossover")
            if cost < solution_archive[i, -1]:
                solution_archive[i, :] = [bagging_alpha_fasts[i], bagging_alpha_slows[i], cost]
        
        print('[Current archive]')
        print(solution_archive)
        
        current_conf_matrix = peak_detector.bagging_records_confusion_matrix(solution_archive, train_records)
        print('Current confusion matrix: [TP,TN,FP,FN]' + str(current_conf_matrix))
        
    # pkl.dump(solution_archive, open("solution_archive.data","wb"))
    train_confusion_matrix = peak_detector.bagging_records_confusion_matrix(solution_archive, train_records)
    test_confusion_matrix = peak_detector.bagging_records_confusion_matrix(solution_archive, test_records)
    
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
    print('Value Error\n')
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
#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and threshold using a confusion matrix-based cost function 

import numpy as np
from ppg_peak_detection import crossover_detector
from ppg_peak_detection import random_search_crossover
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 200 Hz
from time_manager import time

try:
    # Load reference data (44 records for training and 22 for testing)
    # Test data is composed of an equal number of healty and dpoc records
    if len(records) != 66:
        print("Number of records is not 66")
        exit(-1)
        
    train_records = records[11:-11]
    print('Train records: [11:-11], len = ' + str(len(train_records)))
    test_records = records[0:11] + records[-11:]
    print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

    # Sampling frequency
    fs = 200
    # Number of runs to extract stats from
    num_runs = 30
    print('\nNumber of runs = ' + str(num_runs))
    # Number of random search iterations
    num_iterations = 10                                       
    print('Number of iterations = ' + str(num_iterations))
    
    verbosity = True
    
    train_accuracies = []
    test_accuracies = []
    for _ in range(num_runs):
        # Optimize parameters and define model with them
        alpha_cross, alpha_fast, alpha_slow, train_cost = random_search_crossover(train_records, num_iterations, min_alpha = 0.7, max_alpha = 1, verbosity=verbosity)
        peak_detector = crossover_detector(alpha_cross, alpha_fast, alpha_slow, fs)
        
        # Get results for train and test data
        # Train
        # train_cm = peak_detector.literature_record_set_confusion_matrix(train_records)
        # train_precision = train_cm[0] / (train_cm[0] + train_cm[1])
        # train_recall =    train_cm[0] / (train_cm[0] + train_cm[2])
        # train_accuracy = (train_precision + train_recall)/2
        train_accuracy = 1 - train_cost     # Train cost is 1 - acc
        # Test
        test_cm = peak_detector.literature_record_set_confusion_matrix(test_records)
        test_precision = test_cm[0] / (test_cm[0] + test_cm[1])
        test_recall =    test_cm[0] / (test_cm[0] + test_cm[2])
        test_accuracy = (test_precision + test_recall)/2
        test_accuracies.append(test_accuracy)
        
    print(f'Train acc: {np.mean(train_accuracies)} ({np.std(train_accuracies)})')
    print(f'Test acc:  {np.mean(test_accuracies)} ({np.std(test_accuracies)})')
    
    
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
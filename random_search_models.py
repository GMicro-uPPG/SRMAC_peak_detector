#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and threshold using confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from ppg_peak_detection import random_search_crossover
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time

try:
    # Load reference data (44 records for training and 22 for testing)
    # Test data is composed of an equal number of healty and dpoc records
    if len(records) != 66:
        print("Number of records is not 66")
        exit(-1)
        
    print('Train records: [11:-11]')
    train_records = records[11:-11]
    print('Test records: [0:11] u [-11:])')
    test_records = records[0:11] + records[-11:]

    # Random search of alphas, using confusion matrix-based cost
    num_iterations = 1000                                                               # Number of random search iterations
    print('\n Number of iterations = ' + str(num_iterations))
    
    # Optimizes model
    best_solution = random_search_crossover(train_records, num_iterations, min_alpha = 0.9, max_alpha = 1, min_threshold = 0, max_threshold = 1, large_peaks_only=True, verbosity=True)
    
    peak_detector = crossover_detector()
    peak_detector.set_parameters_cross(best_solution[0], best_solution[1], best_solution[2])

    # Get results for train and test data
    # train_confusion_matrix = peak_detector.record_set_confusion_matrix(train_records, "crossover", large_peaks_only = True, peak_len_threshold = best_solution[3])
    # test_confusion_matrix = peak_detector.record_set_confusion_matrix(test_records, "crossover", large_peaks_only = True, peak_len_threshold = best_solution[3])
    # print('\nTrain set confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
    # print('Test set confusion matrix: [TP,TN,FP,FN]' + str(test_confusion_matrix))
    train_cm = peak_detector.literature_record_set_confusion_matrix(train_records, True, 13)
    test_cm = peak_detector.literature_record_set_confusion_matrix(test_records, True, 13)
    print('\nTrain set confusion matrix: [TP,FP,FN]' + str(train_cm))
    print('Test set confusion matrix: [TP,FP,FN]' + str(test_cm))

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
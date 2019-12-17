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
    # Uses 40 records to train model and 20 to test it
    train_records = records[0:40]
    test_records = records[40:60]
    print('records[0:40]')

    # Random search of alphas, using regularized confusion matrix-based cost
    num_iterations = 1000                                                               # Number of random search iterations
    print('\n Number of iterations = ' + str(num_iterations))
    
    # Optimizes model
    best_solution = random_search_crossover(train_records, num_iterations, 0.9, 1, verbosity=True)
    
    peak_detector = crossover_detector()
    peak_detector.set_parameters_cross(best_solution[0], best_solution[1])

    # Get results for train and test data
    train_confusion_matrix = peak_detector.record_set_confusion_matrix(train_records, "crossover")
    test_confusion_matrix = peak_detector.record_set_confusion_matrix(test_records, "crossover")
    print('\nTrain set confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
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
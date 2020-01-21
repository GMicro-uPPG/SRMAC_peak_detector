#!python3
# Author: Victor O. Costa 
# Performs random search on each model's weights, optimizing a confusion matrix metric

import numpy as np
import pickle as pkl
import random
from ppg_peak_detection import crossover_detector
from read_datasets import records                   # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time
from plot import *

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
    
    # Load each member's predictions on train and test data
    train_records_predictions = pkl.load(open("ensemble_train_predictions.pickle","rb"))
    test_records_predictions = pkl.load(open("ensemble_test_predictions.pickle","rb"))
    
    if len(train_records_predictions[0]) != len(test_records_predictions[0]):
        print("Train and test predictions are made with different number of models")
        exit(-1)
    
    ensemble_size = len(train_records_predictions[0])
    
    num_iterations = 10                                        # Number of random search iterations
    print('\nNumber of iterations = ' + str(num_iterations))
    
    # Initial solution is the unweighted voting of the loaded models
    # Compute train accuracies of this initial solution
    peak_detector = crossover_detector()    
    best_weights = np.ones(ensemble_size)
    best_treshold = 0.5
    best_len_thr = 20
    best_cm = peak_detector.ensemble_records_confusion_matrix(train_records, train_records_predictions, best_weights, best_treshold, True, best_len_thr)
    best_accuracy = (best_cm[0] + best_cm[1])/(sum(best_cm))
    print('Initial score: ' + str(best_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(best_cm))
    
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover ensembles
        # Keep ensemble_size best solutions and build voting ensemble by bootstrap sampling (ensemble)
        iteration_weights = np.random.uniform(0, 1, ensemble_size)
        iteration_threshold = np.random.uniform(0, 1)
        iteration_peak_len_threshold = np.random.randint(0, 30)

        # iteration ensemble confusion matrix
        iteration_cm = peak_detector.ensemble_records_confusion_matrix(train_records, train_records_predictions, iteration_weights, iteration_threshold, True, iteration_peak_len_threshold)
        iteration_accuracy = (iteration_cm[0] + iteration_cm[1])/(sum(iteration_cm))
        
        print('(Randomized weights) ' + str(iteration_weights))
        print('(Randomized threshold) ' + str(iteration_threshold))
        print('(Randomized peak len thr) ' + str(iteration_peak_len_threshold))
        print('Score: ' + str(iteration_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(iteration_cm))
        
        if iteration_accuracy > best_accuracy:
            best_weights    = iteration_weights
            best_treshold   = iteration_threshold
            best_len_thr    = iteration_peak_len_threshold
            best_cm         = iteration_cm
            best_accuracy   = iteration_accuracy
        
        print('\n(Current best weights) ' + str(best_weights))
        print('(Current best threshold) ' + str(best_treshold))
        print('(Current best peak len thr) ' + str(best_len_thr))
        print('Score: ' + str(best_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(best_cm))
    
    train_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(train_records, train_records_predictions, best_weights, best_treshold, True, best_len_thr)
    test_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(test_records, test_records_predictions, best_weights, best_treshold, True, best_len_thr)
    
    print('\nTrain set ensemble confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
    print('Test set ensemble confusion matrix: [TP,TN,FP,FN]' + str(test_confusion_matrix))
    
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

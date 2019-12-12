#!python3
# Author: Victor O. Costa 
# Performs random search on each model's weights, optimizing a confusion matrix metric

import numpy as np
import pickle as pkl
import random
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
    train_records = records[0:40]
    test_records = records[40:60]
    
    print('\nrecords[0:40]')

    # Random search of alphas, using regularized confusion matrix-based cost
    peak_detector = crossover_detector()
    # Parameters
    C = 0                                         # Regularization hyperparameter
    print('\nC = ' + str(C))
    num_iterations = 300                            # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))
    
    # Ensemble models
    # ensemble_models[:, 0] <=> best ensemble_alpha_fasts
    # ensemble_models[:, 1] <=> best ensemble_alpha_slows
    # ensemble_models[:, 2] <=> best ensemble_costs
    ensemble_models = pkl.load(open("ensemble_models.pickle", "rb"))
    print(np.array(ensemble_models))
    ensemble_size = len(ensemble_models)
    # Initial solution is the unweighted voting of the loaded models
    # Compute accuracy of this initial solution
    best_weights = (ensemble_size)*[1]
    best_treshold = 0.5
    best_cm = peak_detector.ensemble_records_confusion_matrix(ensemble_models, best_weights, best_treshold, train_records)
    best_accuracy = (best_cm[0] + best_cm[1])/(sum(best_cm))
    print('Initial score: ' + str(best_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(best_cm))
       
    
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover ensembles
        # Keep ensemble_size best solutions and build voting ensemble by bootstrap sampling (ensemble)
        iteration_weights = np.random.uniform(0, 1, ensemble_size)
        iteration_threshold = np.random.uniform(0, 0.1)

        # iteration ensemble confusion matrix
        iteration_cm = peak_detector.ensemble_records_confusion_matrix(ensemble_models, iteration_weights, iteration_threshold, train_records)
        iteration_accuracy = (iteration_cm[0] + iteration_cm[1])/(sum(iteration_cm))
        
        if iteration_accuracy > best_accuracy:
            best_weights    = iteration_weights
            best_treshold   = iteration_threshold
            best_cm         = iteration_cm
            best_accuracy   = iteration_accuracy
        
        print('(Current best weights) ' + str(best_weights))
        print('(Current best threshold) ' + str(best_treshold))
        print('Score: ' + str(best_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(best_cm))
        
    # pkl.dump(ensemble_models, open("ensemble_models.data","wb"))
    train_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(ensemble_models, best_weights, best_treshold, train_records)
    test_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(ensemble_models, best_weights, best_treshold, test_records)
    
    print('Train set ensemble confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
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

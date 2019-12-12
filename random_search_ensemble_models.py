#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and threshold, optimizing a confusion matrix metric

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
    num_iterations = 500                            # Number of random search iterations
    print('\nnum_iterations = ' + str(num_iterations))
    
    # Ensemble models
    # ensemble_models[0] <=> best ensemble_alpha_fasts
    # ensemble_models[1] <=> best ensemble_alpha_slows
    # ensemble_models[2] <=> best ensemble_costs
    ensemble_size = 10
    
    ensemble_models = np.ones((ensemble_size, 3))
    
    ### Ensemble datasets
    train_sampled_records = [0] * ensemble_size
    ## Bootstrap train data for ensemble
    # Generate boostrap (sampling with replacement) fixed indices (ensemble_size x input length matrix)
    #print("Generate bootstrap train records")    
    #bootstrap_indices = np.random.randint(low = 0, high=len(train_records), size=(ensemble_size, len(train_records)))
    # Based on bootstrap indices, create different realizations of the records
    #for i in range(0, ensemble_size):
    #    train_sampled_records[i] = np.array(train_records)[bootstrap_indices[i]]
    
    ## Sample records without replacement (ensemble_size x (ratio * input length matrix))
    sampling_percentage = 0.10 		                
    for i in range(0, ensemble_size):
        train_sampled_records[i] = random.sample(train_records, int(sampling_percentage * len(train_records)))
    
    
    # Define ensemble models initially at zero with infinite costs
    ensemble_models = np.zeros((ensemble_size,3))
    ensemble_models[:, -1] = float('inf')
    # With all the weights equal to 1 and threhsold as 0.5, the voting is unweighted
    models_weights = np.ones(ensemble_size)
    voting_threshold = 0.5
    
    best_solution = [0, 0, ]
    for iteration in range(num_iterations):
        print('\n[Search iteration ' + str(iteration) + ']')
        
        ## Optimize crossover ensembles
        # Keep ensemble_size best solutions and build voting ensemble by bootstrap sampling (ensemble)
        ensemble_alpha_fasts = np.random.uniform(0.9, 1, ensemble_size)
        ensemble_alpha_slows = np.random.uniform(ensemble_alpha_fasts, ensemble_size*[1], ensemble_size)

        for i in range(0, ensemble_size):
            # print("Ensemble member " + str(i))
            peak_detector.set_parameters_cross(ensemble_alpha_fasts[i], ensemble_alpha_slows[i])

            cost = peak_detector.total_regularized_cost(train_sampled_records[i], C, "crossover")
            if cost < ensemble_models[i, -1]:
                ensemble_models[i, :] = [ensemble_alpha_fasts[i], ensemble_alpha_slows[i], cost]
        
        print('[Current models, individual costs]')
        print(ensemble_models)
        
        # Current ensemble confusion matrix
        current_cm = peak_detector.ensemble_records_confusion_matrix(ensemble_models, models_weights, voting_threshold, train_records)
        current_accuracy = (current_cm[0] + current_cm[1])/(sum(current_cm))
        print('(Current ensemble) Score: ' + str(current_accuracy) + ', Matrix [TP,TN,FP,FN]' + str(current_cm))
        
    # pkl.dump(ensemble_models, open("ensemble_models.data","wb"))
    train_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(ensemble_models, models_weights, voting_threshold, train_records)
    test_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(ensemble_models, models_weights, voting_threshold, test_records)
    
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

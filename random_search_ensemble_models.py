#!python3
# Author: Victor O. Costa 
# Performs random search on the crossover's alphas and threshold, optimizing a confusion matrix metric

import numpy as np
import pickle as pkl
import random
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
    
    num_iterations = 1000                            # Number of random search iterations
    ensemble_size = 10
    
    print('\nIterations per model = ' + str(num_iterations))
    print('\nEnsemble size = ' + str(ensemble_size))
    
    # Percentage of train data sampling for each model's optimization
    sampling_percentage = 0.10
    print('\nSampling percentage = ' + str(sampling_percentage))
    # Optimize each model for its own subset of the train records
    ensemble_models = []
    for model_index in range(ensemble_size):
        # Sample records without replacement (ensemble_size x (sampling_percentage * input length matrix))
        sampled_train_records = random.sample(train_records, int(sampling_percentage * len(train_records)))
        
        # Random search of model's alphas over the saampled train records
        print("\nSearch for model " + str(model_index))
        model_parameters = random_search_crossover(sampled_train_records, num_iterations, min_alpha = 0.9, max_alpha = 1, min_threshold = 0, max_threshold = 1, large_peaks_only=True, verbosity=False)
        print("Model parameters: " + str(model_parameters[0:-1]))
        
        # Print costs on partition and train set
        detector = crossover_detector()
        detector.set_parameters_cross(model_parameters[0], model_parameters[1], model_parameters[2])
        train_cm = detector.record_set_confusion_matrix(train_records, "crossover", True, model_parameters[3])
        # Cost = 1 - acc
        train_cost = 1 - (train_cm[0]+train_cm[1])/(sum(train_cm))
        
        print("Cost on partition: " + str(model_parameters[-1]) + " / Cost on train set: " + str(train_cost))
        ensemble_models.append(model_parameters)
        
    # Save search result
    pkl.dump(ensemble_models, open("ensemble_models.pickle","wb"))
    # ensemble_models = pkl.load(open("ensemble_models.pickle","rb"))
        
    ## Extract predictions from all models for all records on train and test data, record-wise
    train_records_predictions = []
    for record in train_records:
        # For a specific record, get the predictions of all ensemble members
        ppg_signal = record.ppg[1]
        single_record_predictions = []
        for model_parameters in ensemble_models:
            # Specify model with the given parameters
            model = crossover_detector()
            model.set_parameters_cross(model_parameters[0], model_parameters[1], model_parameters[2])
            # Append this model's predictions over the given record
            single_record_predictions.append( model.detect_peaks_cross(ppg_signal)[-1] )
        # Append predictions from all models over the given record
        train_records_predictions.append(single_record_predictions)
    
    test_records_predictions = []
    for record in test_records:
        # For a specific record, get the predictions of all ensemble members
        ppg_signal = record.ppg[1]
        single_record_predictions = []
        for model_parameters in ensemble_models:
            # Specify model with the given parameters
            model = crossover_detector()
            model.set_parameters_cross(model_parameters[0], model_parameters[1], model_parameters[2])
            # Append this model's predictions over the given record
            single_record_predictions.append( model.detect_peaks_cross(ppg_signal)[-1] )
        # Append predictions from all models over the given record
        test_records_predictions.append(single_record_predictions)
    
    # Save ensemble train and test predictions
    pkl.dump(train_records_predictions, open("ensemble_train_predictions.pickle","wb"))
    pkl.dump(test_records_predictions, open("ensemble_test_predictions.pickle","wb"))
    
    # With all the weights equal to 1 and threhsold as 0.5, the voting is unweighted
    models_weights = np.ones(ensemble_size)
    voting_threshold = 0.5
    peak_detector = crossover_detector()
    train_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(train_records, train_records_predictions, models_weights, voting_threshold, True, 20)
    test_confusion_matrix = peak_detector.ensemble_records_confusion_matrix(test_records, test_records_predictions, models_weights, voting_threshold, True, 20)
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

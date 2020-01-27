#!python3
# Author: Victor O. Costa

import numpy as np
import pickle as pkl
import random
from ppg_peak_detection import crossover_detector
from read_datasets import records # This will load 60 records (o to 59). Rercord sample rate = 125Hz
from time_manager import time

# Load reference data (44 records for training and 22 for testing)
# Test data is composed of an equal number of healty and dpoc records
if len(records) != 66:
    print("Number of records is not 66")
    exit(-1)
    
print('\nTrain records: [11:-11]')
train_records = records[11:-11]
print('Test records: [0:11] u [-11:])')
test_records = records[0:11] + records[-11:]


# Number of objective function evaluations is num_iter * (num_models + 1) + 1
# Number of iterations
num_iter = 1000

## Ensemble parameters
num_models = 5

print("\nNumber of models: " + str(num_models))
print("Number of iteratios: " + str(num_iter))

# Crossovers alphas
min_alphas = 0.9 
max_alphas = 1.0

# Crossover percentage threshold
min_percentage_thr = 0.0
max_percentage_thr = 1.0

# Short peak discard threshold
min_peak_thr = 0
max_peak_thr = 30

if (min_alphas < 0) or (min_alphas > 1) or (max_alphas < 0) or (max_alphas > 1):
    print("Minimum and maximum alphas must be between 0 and 1")
    exit(-1)
    
## ENSEMBLE RANDOM INITIALIZATION
current_solution= { "ensemble_models":[[] for x in range(num_models)], "peak_len_thr" : 0,
                    "train_predictions" : [[[] for y in range(num_models)] for x in range(len(train_records))],
                    "confusion_matrix": []}
detector = crossover_detector()
for model_index in range(num_models):
    # Randomize crossover parameters
    alpha_fast = np.random.uniform(min_alphas, max_alphas)
    alpha_slow = np.random.uniform(alpha_fast, max_alphas)
    percentage_threshold = np.random.uniform(min_percentage_thr, max_percentage_thr)
    detector.set_parameters_cross(alpha_fast, alpha_slow, percentage_threshold)
    
    # Keep models in the solution dictionary
    current_solution["ensemble_models"][model_index] = [alpha_fast, alpha_slow, percentage_threshold]
    
    # Get each model's predictions on train records 
    for record_index, record in enumerate(train_records):
        ppg_signal = record.ppg[1]
        current_solution["train_predictions"][record_index][model_index] = detector.detect_peaks_cross(ppg_signal)[-1]
        
# Randomize peak length threshold
current_solution["peak_len_thr"] = np.random.randint(min_peak_thr, max_peak_thr)

# Combine predictions of all models, using unweighted majority voting and discarding short peaks, and compute the resulting confusion matrix       
current_solution["confusion_matrix"] = detector.ensemble_records_confusion_matrix(train_records, current_solution["train_predictions"], [1]*num_models, 0.5, True, current_solution["peak_len_thr"])
current_accuracy = (current_solution["confusion_matrix"][0] + current_solution["confusion_matrix"][1])/(sum(current_solution["confusion_matrix"] ))

print('\n[Initial ensemble]')
print('\nCrossover models')
print(np.array(current_solution["ensemble_models"]))
print('\nPeak length threshold: ' + str(current_solution["peak_len_thr"]))
#print('\n\n' + str(np.shape(models_train_predictions)))
       
print('\nTrain set confusion matrix: [TP,TN,FP,FN]' + str(current_solution["confusion_matrix"] ))
print('Train set accuracy = ' + str(current_accuracy) + '\n')

print("\nOptimization loop\n")
## Optimization loop
for iteration in range(num_iter):
    print("[Iteration " + str(iteration) + ']')
    # Greedy optimization for each model
    for model_index in range(num_models):
        # Randomize crossover parameters
        alpha_fast = np.random.uniform(min_alphas, max_alphas)
        alpha_slow = np.random.uniform(alpha_fast, max_alphas)
        percentage_threshold = np.random.uniform(min_percentage_thr, max_percentage_thr)
        detector.set_parameters_cross(alpha_fast, alpha_slow, percentage_threshold)
        print("Randomized model " + str(model_index) + ": " + str([alpha_fast, alpha_slow, percentage_threshold]))
        
        # Get randomized model predictions on train records
        # Iteration predictions are the same as current predictions except for the model being randomized
        model_predictions = []
        for record_index, record in enumerate(train_records):
            ppg_signal = record.ppg[1]
            model_predictions.append( detector.detect_peaks_cross(ppg_signal)[-1] )
        
        iteration_ensemble_predictions = list(current_solution["train_predictions"])
        for record_index in range(len(iteration_ensemble_predictions)):
            iteration_ensemble_predictions[record_index][model_index] = model_predictions[record_index]
        
        #print(np.shape(current_solution["train_predictions"]))
        #print(np.shape(model_predictions))  
        #print(iteration_ensemble_predictions == current_solution["train_predictions"])
        
        # Get accuracy of the current ensemble together with randomized model
        iteration_confusion_matrix = detector.ensemble_records_confusion_matrix(train_records, iteration_ensemble_predictions, [1]*num_models, 0.5, True, current_solution["peak_len_thr"])
        iteration_accuracy = (iteration_confusion_matrix[0] + iteration_confusion_matrix[1])/(sum( iteration_confusion_matrix ))
        
        print('Current accuracy = ' + str(current_accuracy))
        print('Accuracy using randomized model = ' + str(iteration_accuracy) + '\n')
        
        if iteration_accuracy > current_accuracy:
            current_solution["ensemble_models"][model_index] = [alpha_fast, alpha_slow, percentage_threshold]
            current_solution["train_predictions"][:][model_index] = model_predictions
            current_solution["confusion_matrix"] = iteration_confusion_matrix
            current_accuracy = iteration_accuracy
    
    # Randomize peak length threshold and get accuracy
    iteration_peak_len_thr = np.random.randint(min_peak_thr, max_peak_thr)
    iteration_confusion_matrix = detector.ensemble_records_confusion_matrix(train_records, current_solution["train_predictions"], [1]*num_models, 0.5, True, iteration_peak_len_thr)
    iteration_accuracy = (iteration_confusion_matrix[0] + iteration_confusion_matrix[1])/(sum( iteration_confusion_matrix ))
    
    print('Current accuracy = ' + str(current_accuracy))
    print('Accuracy using length threshold = ' + str(iteration_accuracy) + '\n')
    
    if iteration_accuracy > current_accuracy:
        current_solution["peak_len_thr"] = iteration_peak_len_thr
        current_solution["confusion_matrix"] = iteration_confusion_matrix
        current_accuracy = iteration_accuracy

        
print('\n[Final ensemble]')
print('\nCrossover models')
print(np.array(current_solution["ensemble_models"]))
print('\nPeak length threshold: ' + str(current_solution["peak_len_thr"]))
       
print('\nTrain set confusion matrix: [TP,TN,FP,FN]' + str(current_solution["confusion_matrix"] ))
print('Train set accuracy = ' + str(current_accuracy) + '\n')

# Extract test set results
test_predictions = [[[] for y in range(num_models)] for x in range(len(test_records))]
for model_index, model_parameters in enumerate(current_solution["ensemble_models"]):
    # Define model based on found parameters
    detector.set_parameters_cross(model_parameters[0], model_parameters[1], model_parameters[2])
    for record_index, record in enumerate(test_records):
        ppg_signal = record.ppg[1]
        test_predictions[record_index][model_index] = detector.detect_peaks_cross(ppg_signal)[-1]

test_confusion_matrix = detector.ensemble_records_confusion_matrix(test_records, test_predictions, [1]*num_models, 0.5, True, current_solution["peak_len_thr"])
test_accuracy = (test_confusion_matrix[0] + test_confusion_matrix[1])/(sum(test_confusion_matrix))
print('\nTest set confusion matrix: [TP,TN,FP,FN]' + str(test_confusion_matrix))
print('Test set accuracy = ' + str(test_accuracy) + '\n')
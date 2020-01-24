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

## Ensemble parameters
num_models = 5

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
    
## Initialize ensemble randomly, 
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
print('\nTrain set accuracy = ' + str(current_accuracy))






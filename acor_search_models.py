#!python3
# Author: Victor O. Costa 
# Performs optimization on the crossover's alphas and derivtive threshold via ant colony for continuous domains using confusion matrix-based cost function 

import numpy as np
import pickle as pkl
from ppg_peak_detection import crossover_detector
from ant_colony_for_continuous_domains import ACOr
from read_datasets import records
from time_manager import time

# Given a set of records, defines a method to obtain the cost of crossover alphas
def generic_crossover_cost(detector, records):
    def crossover_cost(alphas):
        alpha_fast = alphas[0]
        alpha_slow = alphas[1]
        
        detector.set_parameters_cross(alpha_fast, alpha_slow)
        tp, tn, fp, fn = detector.record_set_confusion_matrix(records, "crossover")
        accuracy = float(tp + tn) / float(tp + tn + fp + fn)
        cost = 1 - accuracy
        
        return cost
    return crossover_cost
    
# Use 40 records to train model and 20 to test it
train_records = records[0:40]
test_records = records[40:60]
print('\nrecords[0:40]')

# Search parameters
num_iterations = 100
archive_size = 20
colony_size = 2
search_locality = 0.7
speed_of_convergence = 0.85

alphas_ranges = [[0.9, 1],
                 [0.9, 1]]

print('num_iterations = ' + str(num_iterations))
print('archive_size = ' + str(archive_size))
print('colony_size = ' + str(colony_size))
print('search_locality = ' + str(search_locality))
print('speed_of_convergence = ' + str(speed_of_convergence))

# Optimization
peak_detector = crossover_detector()

colony = ACOr()
colony.set_cost(generic_crossover_cost(peak_detector, train_records))
colony.set_parameters(num_iterations, colony_size, archive_size, search_locality, speed_of_convergence)
colony.set_variables(2, alphas_ranges)
solution_archive = colony.optimize()
best_solution = solution_archive[0,:]

print("\nFinal Solution archive:")
print(solution_archive)

# Results for train and test sets
peak_detector.set_parameters_cross(best_solution[0], best_solution[1])
# peak_detector.set_parameters_var(best_solution[0], best_solution[1], best_solution[2])
# peak_detector.set_parameters_mix(best_solution[0], best_solution[1], best_solution[2], best_solution[3], best_solution[4])

train_confusion_matrix = peak_detector.record_set_confusion_matrix(train_records, "crossover")
test_confusion_matrix = peak_detector.record_set_confusion_matrix(test_records, "crossover")

print('\nTrain set confusion matrix: [TP,TN,FP,FN]' + str(train_confusion_matrix))
print('Test set confusion matrix: [TP,TN,FP,FN]' + str(test_confusion_matrix))


print('\nLast timestamp: ' + str(time.getTimestamp()))
print('Last time: ' + str(time.getTime()))

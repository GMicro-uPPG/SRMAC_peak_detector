#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
from read_datasets import records

if len(records) != 66:
        print("Number of records is not 66")
        exit(-1)
        
train_records = records[11:-11]
print('Train records: [11:-11], len = ' + str(len(train_records)))
test_records = records[0:11] + records[-11:]
print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

peak_detector = crossover_detector()
peak_detector.set_parameters_cross(alpha_crossover = 0.8705192717851324, alpha_fast = 0.903170529094925 , alpha_slow = 0.9586798163470)               

train_cm = peak_detector.literature_record_set_confusion_matrix(train_records)
test_cm = peak_detector.literature_record_set_confusion_matrix(test_records)
print('\nTrain set confusion matrix: [TP,FP,FN]' + str(train_cm))
print('Test set confusion matrix: [TP,FP,FN]' + str(test_cm))

print('\n\nTERMA')
TERMA_train_cm = peak_detector.terma_record_set_confusion_matrix(train_records)
TERMA_test_cm = peak_detector.terma_record_set_confusion_matrix(test_records)
print('\nTrain set confusion matrix: [TP,FP,FN]' + str(TERMA_train_cm))
print('Test set confusion matrix: [TP,FP,FN]' + str(TERMA_test_cm))
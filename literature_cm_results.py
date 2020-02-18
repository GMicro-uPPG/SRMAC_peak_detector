#!python3

import numpy as np
from ppg_peak_detection import crossover_detector
from read_datasets import records

if len(records) != 66:
        print("Number of records is not 66")
        exit(-1)
        
print('Train records: [11:-11]')
train_records = records[11:-11]
print('Test records: [0:11] u [-11:])')
test_records = records[0:11] + records[-11:]

peak_detector = crossover_detector()
peak_detector.set_parameters_cross(0.9194304850895123 , 0.9387829409026597, 0.35405603150462084)
train_cm = peak_detector.literature_record_set_confusion_matrix(train_records, True, 13)
test_cm = peak_detector.literature_record_set_confusion_matrix(test_records, True, 13)
print('\nTrain set confusion matrix: [TP,FP,FN]' + str(train_cm))
print('Test set confusion matrix: [TP,FP,FN]' + str(test_cm))
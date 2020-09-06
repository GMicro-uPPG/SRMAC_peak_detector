#!python3

# MIT License

# Copyright (c) 2016 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Own
from read_datasets import records
from ppg_peak_detection import crossover_detector
# Third party
import numpy as np

if len(records) != 66:
    print('Number of records is not 66')
    exit(-1)
        
train_records = records[11:-11]
print('Train records: [11:-11], len = ' + str(len(train_records)))
test_records = records[0:11] + records[-11:]
print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

peak_detector = crossover_detector(0.8705192717851324, 0.903170529094925, 0.9586798163470798)

train_cm = peak_detector.literature_record_set_confusion_matrix(train_records)
test_cm = peak_detector.literature_record_set_confusion_matrix(test_records)
print('\nTrain set confusion matrix: [TP,FP,FN]' + str(train_cm))
print('Test set confusion matrix: [TP,FP,FN]' + str(test_cm))

print('\n\nTERMA')
TERMA_train_cm = peak_detector.terma_record_set_confusion_matrix(train_records)
TERMA_test_cm = peak_detector.terma_record_set_confusion_matrix(test_records)
print('\nTrain set confusion matrix: [TP,FP,FN]' + str(TERMA_train_cm))
print('Test set confusion matrix: [TP,FP,FN]' + str(TERMA_test_cm))
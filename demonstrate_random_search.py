#!python3

# MIT License

# Copyright (c) 2021 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

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

# Author: Victor O. Costa

# Application modules
from crossover_detector import crossover_detector
from read_datasets import records # This will load 66 records. Rercord sample rate = 200 Hz
import utilities
import optimization
# Third party
import numpy as np
import time_manager

try:
    # Load reference data (44 records for training and 22 for testing)
    # Test data is composed of an equal number of healty and dpoc records
    if len(records) != 66:
        print('Number of records is not 66')
        exit(-1)
        
    train_records = records[11:-11]
    print('Train records: [11:-11], len = ' + str(len(train_records)))
    test_records = records[0:11] + records[-11:]
    print('Test records: [0:11] u [-11:]), len = ' + str(len(test_records)))

    # Sampling frequency
    Fs = 200
    # Number of runs to extract stats from
    num_runs = 3
    print('\nNumber of runs = ' + str(num_runs))
    # Iterations of interest for random search
    iterations_of_interest = [2,5]                                       
    print('Iterations of interest = ' + str(iterations_of_interest))
    
    verbosity = True
    # Train acc histories
    hist_train_accs = []
    hist_test_accs = []
    for _ in range(num_runs):
        # Get history of solutions defined by iterations of interest
        solutions_of_interest = optimization.random_search_crossover(train_records = train_records,
                                iterations_of_interest = iterations_of_interest, min_alpha = 0.7, max_alpha = 1,
                                sampling_frequency=Fs, verbosity=verbosity)
        run_train_accuracies = []
        run_test_accuracies = []
        # For each solution define a model and extract test acc
        for soi in solutions_of_interest:
            alpha_cross, alpha_fast, alpha_slow, train_cost = soi
            peak_detector = crossover_detector(alpha_cross, alpha_fast, alpha_slow)            # Get results for train and test data
            # Train
            train_accuracy = 1 - train_cost     # Train cost is 1 - acc
            run_train_accuracies.append(train_accuracy)
            # Test
            test_cm = utilities.record_set_confusion_matrix(peak_detector, test_records, Fs)
            test_precision = test_cm[0] / (test_cm[0] + test_cm[1])
            test_recall =    test_cm[0] / (test_cm[0] + test_cm[2])
            test_accuracy = (test_precision + test_recall)/2
            run_test_accuracies.append(test_accuracy)
        hist_train_accs.append(list(run_train_accuracies))
        hist_test_accs.append(list(run_test_accuracies))
    
    print('Train accuracies history')
    print(hist_train_accs)
    print('Test accuracies history')
    print(hist_test_accs)
    # print(f'Train acc: {np.mean(train_accuracies)} ({np.std(train_accuracies)})')
    # print(f'Test acc:  {np.mean(test_accuracies)} ({np.std(test_accuracies)})')
    
    
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))


except IOError:
    print('Error: An error occurred trying to read the file.\n')
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
except ValueError:
    print('Error: Non-numeric data found in the file.\n')
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
except utilitiesError:
    print('Error: No module found.\n')
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
except EOFError:
    print('Error: Why did you do an EOF on me?\n')
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
except KeyboardInterrupt:
    print('Error: You cancelled the operation.\n')
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
except Exception as e:
    print('An error occurred:', e)
    print('\nLast timestamp: ' + str(time_manager.time.getTimestamp()))
    print('Last time: ' + str(time_manager.time.getTime()))
#/except
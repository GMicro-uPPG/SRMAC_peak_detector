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

# Third party
import numpy as np
# Application modules
from crossover_detector import crossover_detector

def random_search_crossover(train_records, iterations_of_interest, min_alpha, max_alpha, sampling_frequency, verbosity):
    ''' Given the number of iterations and alphas range,
        performs random search on the crossover's alphas using train data accuracy as fitness metric. '''
    if (min_alpha < 0) or (min_alpha > 1) or (max_alpha < 0) or (max_alpha > 1):
        print('Error, minimum and maximum alphas must be between 0 and 1')
        exit(-1)
    if len(iterations_of_interest) == 0:
        print('Error, iterations of interest must not be empty')
        exit(-1)
    if np.min(iterations_of_interest) <= 0 : 
        print('Error, the minimum iteration of interest must be 1')
        exit(-1)
    if verbosity != False and verbosity != True:
        print('Error, erbosity must be boolean')
        exit(-1)
    
    num_iterations = int(np.max(iterations_of_interest))
    
    # The initial solution has infinite cost, and therefore any solution is better than the initial one
    best_solution = [0, 0, 0, float('inf')]
    solutions_of_interest = []
    
    # Optimization loop
    for iteration in range(num_iterations):
        if verbosity == True: print('\n[Search iteration ' + str(iteration) + ']')

        # Slow alpha depends on fast alpha (fast alpha < slow alpha)
        alpha_fast = np.random.uniform(min_alpha, max_alpha)
        alpha_slow = np.random.uniform(alpha_fast, max_alpha)
        # The crossover alpha is independent of fast and slow alphas
        alpha_crossover = np.random.uniform(min_alpha, max_alpha)
        peak_detector = crossover_detector(alpha_crossover, alpha_fast, alpha_slow)
        
        # Run the detector defined above in the train records and extract SE and P+
        tp, fp, fn = record_set_confusion_matrix(peak_detector, train_records, sampling_frequency)
        
        SE = tp / (tp + fn)
        Pp = tp / (tp + fp)
        cost = 1 - (SE + Pp)/2
        
        if cost < best_solution[-1]:
            best_solution = [alpha_crossover, alpha_fast, alpha_slow, cost]
        
        # Store current best solution in iterations of interest
        if iteration in (np.array(iterations_of_interest) - 1):
            solutions_of_interest.append(list(best_solution))
            
        if verbosity == True:
            print('Alphas: crossover, fast, slow : cost')
            print(f'[{iteration}] {alpha_crossover}, {alpha_fast}, {alpha_slow} : {cost}')
            print(f'[best] {best_solution[0]} {best_solution[1]} {best_solution[2]} : {best_solution[-1]}')
    
    return solutions_of_interest 

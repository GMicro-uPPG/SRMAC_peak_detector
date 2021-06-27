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

import numpy as np
from ppg_peak_detection import crossover_detector

def peak_positions(ppg_signal, detected_peaks):
    ''' From detected peak regions, extract exact peak locations '''
    if len(ppg_signal) != len(detected_peaks):
        print('PPG signal and detected peaks do not match')
        exit()
        
    # Aux. to determine edges in the detected peak regions
    state = detected_peaks[0]               
    # State defines initial value for flag used in sample comparison with the current peak
    if state == 0:                  
        flag_max = False                    
    else: 
        flag_max = True
    
    current_peak = 0
    current_location = 0
    
    positions = []
    for i in range(len(ppg_signal)):
        # Check for edges to decide when to check for peaks and when to store them 
        if state != detected_peaks[i]:
            # Rising edge 
            if state == 0:
                # keep first region sample (magnitude and location)
                current_peak = ppg_signal[i]
                current_location = i
                # signal to start updating the current peak
                flag_max = True
            # Falling edge
            else:
                positions.append(current_location)
                # signal to stop updating the current peak
                flag_max = False
                
            state = detected_peaks[i]
                
        # Find peak locations when 'detected_peaks' is high (1-valued)
        if flag_max and ppg_signal[i] > current_peak:
            current_peak = ppg_signal[i]
            current_location = i
    
    # Include possible last peak not detected because falling edge has not occurred
    if flag_max:
        positions.append(current_location)
    
    return positions
    
def signal_confusion_matrix(detected_locations, reference_locations, sampling_frequency):
    ''' The confusion (triangular) matrix defined in the literature considers if a peak was detected in the
        neighborhood of a reference peak, and has no definition of true negatives '''
    
    # running through the reference peaks one can extract true positives and false negatives
    # false positives = all detected - true positives
    
    # Sampling interval in seconds
    T = 1/sampling_frequency
    # Neighborhood definition in seconds
    neighborhood_time = 0.1
    # Neighborhood definition in number of samples
    neighborhood_samples = int(neighborhood_time / T)
    
    true_positives = 0 
    false_positives = 0
    false_negatives = 0
            
    for reference in reference_locations:
        # Detection of the reference peak is initially assumed to be false
        correct_detection = False
        for detected in detected_locations:
            if detected > (reference - neighborhood_samples) and detected < (reference + neighborhood_samples):
                correct_detection = True
        
        if correct_detection:
            true_positives += 1
        else:
            false_negatives += 1
            
    false_positives = len(detected_locations) - true_positives
    
    return true_positives, false_positives, false_negatives    

def record_set_confusion_matrix(peak_detector, ppg_records, sampling_frequency):
    ''' Given a peak detector and a set of records containing ppg signals and peak references,
        returns the confusion (triangular) matrix based on the literature.'''

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for index, record in enumerate(ppg_records):
        #print('Cost calculation for record ', index)
        ppg_signal = record.ppg[1]
        reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
        # Detect peaks using the 
        _, _, _, detected_peaks = peak_detector.get_peak_blocks(ppg_signal)
        detected_locations = peak_positions(ppg_signal, detected_peaks)
        
        # Get record's confusion matrix and regularization term
        tp, fp, fn = signal_confusion_matrix(detected_locations, reference_peaks, sampling_frequency)
        true_positives += tp; false_positives += fp; false_negatives += fn
        
    return true_positives, false_positives, false_negatives
 
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
        peak_detector = crossover_detector(alpha_crossover, alpha_fast, alpha_slow, sampling_frequency)
        
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
    
def calculate_heart_rates(peaks_array, freq):
    ''' Use the beats to calculate Heart Rates in bpm '''
    sampling_t = 1.0/freq
    heart_rates = []
    
    samples_interval = 0
    old_state = 0
    for index, state in enumerate(peaks_array):
        if state != old_state and state != 0:
            # 1/(samples/beat * secs/sample) = beats/sec
            if index == 0:
                samples_interval = 1
            hr = 60.0/(samples_interval * sampling_t)       # heart rates in bpm
            heart_rates.append(hr)
            samples_interval = 0
        else:
            samples_interval += 1    
        old_state = state
        
    return heart_rates

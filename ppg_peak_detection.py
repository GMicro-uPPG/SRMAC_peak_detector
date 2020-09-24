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

# Python std library
import sys
# Third party
import numpy as np
import scipy

class crossover_detector:
    ''' Class to process the PPG signal and indicate peaks using a crossover of moving averages '''
    
    def __init__(self, alpha_crossover:float, alpha_fast:float, alpha_slow:float, sampling_frequency:float):
        ''' Constructor '''
        
        # Sanity check
        if any([(alpha >= 1) or (alpha <= 0) for alpha in [alpha_crossover, alpha_fast, alpha_slow]]):
            print('Error, alphas must be in 0 <= a <= 1')
            exit(-1)
        if sampling_frequency < 0:
            print('Sampling frequency must be positive')
            exit(-1)
            
        # Smoothing constants
        self.alpha_crossover =  alpha_crossover 
        self.alpha_fast =       alpha_fast
        self.alpha_slow =       alpha_slow
        # Sampling frequency
        self.fs = sampling_frequency
        
        # Variables
        self.average_fast = 0.0
        self.average_slow = 0.0
        self.crossover_index = 0.0
        
    def exponential_ma(self, alpha, input_value, old_average):
        ''' Returns exponential moving average without internal memory '''
        return (1-alpha)*input_value + alpha*old_average

    def reset_state(self, first_ppg_val):
        ''' Resets fast and slow averages '''
        self.average_fast = first_ppg_val
        self.average_slow = first_ppg_val
        self.crossover_index = 0.0
              
    def detect_peaks_cross(self, raw_ppg):
        ''' Crossover based peak detection  '''
        self.reset_state(raw_ppg[0])
        
        fast_averages = []
        slow_averages = []
        crossover_indices = []
        peaks_array = []
        
        # Low-pass filter parameters
        order = 2
        nyquist_freq = 0.5 * self.fs                # 
        freq_cut_hz = 8                             # Analog cutoff frequency
        freq_cut_norm = freq_cut_hz / nyquist_freq   # Digital (normalized) cutoff frequency
        # Butterworth LP as cascaded biquads
        sos = scipy.signal.iirfilter(N=order, Wn=freq_cut_norm, btype='lowpass', analog=False, ftype='butter', output='sos')
        filtered_ppg = scipy.signal.sosfilt(sos, raw_ppg)
        
        for index, ppg_value in enumerate(filtered_ppg):
            # Update model
            self.average_fast       = self.exponential_ma(self.alpha_fast       , ppg_value, self.average_fast)
            self.average_slow       = self.exponential_ma(self.alpha_slow       , ppg_value, self.average_slow)
            self.crossover_index    = self.exponential_ma(self.alpha_crossover  , self.average_fast - self.average_slow, self.crossover_index)
            
            fast_averages.append(self.average_fast)
            slow_averages.append(self.average_slow)
            crossover_indices.append(self.crossover_index)
            
            if self.crossover_index > 0:
                peaks_array.append(1)
            else:  
                peaks_array.append(0)
            
        return fast_averages, slow_averages, crossover_indices, peaks_array
        
    # def ignore_short_peaks(self, detected_peaks, peak_len_threshold):
        # ''' Ignore peaks with low duration  '''
        # peaks_array = list(detected_peaks)
        # peak_len_counter = 0
        # for index, peak_state in enumerate(peaks_array):
            # if peak_state == 1:
                # peak_len_counter += 1
            # else:
                # if peak_len_counter < peak_len_threshold and peak_len_counter > 0:
                    # peaks_array[(index - peak_len_counter) : index] = [0]*peak_len_counter
                # peak_len_counter = 0    
        
        # return peaks_array
        
    def peak_positions(self, ppg_signal, detected_peaks):
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
        
        peak_positions = []
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
                    peak_positions.append(current_location)
                    # signal to stop updating the current peak
                    flag_max = False
                    
                state = detected_peaks[i]
                    
            # Find peak locations when 'detected_peaks' is high (1-valued)
            if flag_max and ppg_signal[i] > current_peak:
                current_peak = ppg_signal[i]
                current_location = i
        
        # Include possible last peak not detected because falling edge did not occurred
        if flag_max:
            peak_positions.append(current_location)
        
        return peak_positions
        
    def calculate_heart_rates(self, peaks_array, freq):
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
        
    def custom_signal_confusion_matrix(self, detected_peaks, peaks_reference):
        ''' Given a set of detected peaks and peaks reference, returns the confusion matrix.'''
        # Our confusion matrix, not used in the literature
        
        true_positives = 0 
        true_negatives = 0 
        false_positives = 0
        false_negatives = 0
        
        fp_hill_flag = False
        
        # Assumes that the detected_peaks array treats 0 as negative prediction and 1 as positive prediction
        state_peaks = 0                                                 # Number of peaks for a given state 
        ref_index = 0                                                   # Index of the reference peaks array
        tn_flag = False                                                 # Flag to avoid a falsely detected peak in one valley to generate two true negatives for that valley 
        confusion_array = []                                            # Array to keep which positions corresponds to which CM members
        for index, prediction in enumerate(detected_peaks):
            if index == 0:
                state = prediction
            # Updates confusion matrix when reaches an edge 
            if prediction != state:
                # Rising edge
                if state == 0:
                    # For no reference peaks in a prediction valley, increment true negatives by one
                    if state_peaks == 0:
                        confusion_array.append(('tn',index))
                        #print('True negative at index = ', index) 
                        true_negatives += 1
                    # For one or more reference peaks in a prediction valley, consider the false negatives and true negatives around it
                    else:
                        confusion_array.append(('fn',index))
                        #print('False negative at index = ', index) 
                        false_negatives += state_peaks
                        true_negatives += state_peaks + 1
                
                # Falling edge
                elif state == 1:
                    # For no reference peaks in a prediction hill, increments false positives.
                    if state_peaks == 0:
                        confusion_array.append(('fp',index))
                        #print('False positive at index = ', index) 
                        if not fp_hill_flag:
                            false_positives += 1
                        # If the false positive is preceded by a true negative, it means that the previous and next true negatives must be ignored
                        true_negatives -= 1
                        fp_hill_flag = True
                    # For more than one reference peaks in a prediction hill, increments the true positives and the false positives with reference to the reference valleys between ref. peaks 
                    else:
                        confusion_array.append(('tp',index))
                        #print('True positive at index = ', index) 
                        true_positives += state_peaks
                        false_positives += state_peaks - 1
                        if fp_hill_flag:
                            true_negatives -= 1
                            
                            fp_hill_flag = False
                    
                state_peaks = 0
                state = prediction
                
            # Checks if a peak ocurred at the current position
            if index == peaks_reference[ref_index]:
                state_peaks += 1
                if ref_index < len(peaks_reference) - 1:
                    ref_index += 1
            
        # Updates confusion matrix in the end of the signal
        # if index == len(detected_peaks) - 1:
            
        if state == 0:
            if state_peaks == 0:
                confusion_array.append(('tn',index))
                true_negatives += 1
            else:
                confusion_array.append(('fn',index))
                false_negatives += state_peaks
                true_negatives += state_peaks + 1
                
        elif state == 1:
            if state_peaks == 0:
                confusion_array.append(('fp',index)) 
                false_positives += 1
            # For more than one reference peaks in a prediction hill, increments the true positives and the false positives with reference to the reference valleys between ref. peaks 
            else:
                confusion_array.append(('tp',index))
                true_positives += state_peaks
                false_positives += state_peaks - 1
                        
        return true_positives, true_negatives, false_positives, false_negatives, confusion_array
    
    def custom_record_set_confusion_matrix(self, ppg_records, method):
        ''' Given a set of records containing ppg signals and peak references, returns the confusion matrix.'''
        
        if method != 'crossover' and method != 'variance' and method != 'mix':
            print('Please choose a valid method')
            exit(-1)
        
        true_positives = 0 
        true_negatives = 0 
        false_positives = 0
        false_negatives = 0
        
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Detect peaks using current set of parameters
            _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
            # _, _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
        
        # Get record's confusion matrix and regularization term
        tp, tn, fp, fn, _ = self.custom_signal_confusion_matrix(detected_peaks, reference_peaks)
        true_positives += tp; true_negatives += tn; false_positives += fp; false_negatives += fn
                    
        return true_positives, true_negatives, false_positives, false_negatives
    
    def literature_signal_confusion_matrix(self, detected_locations, reference_locations):
        ''' The confusion (triangular) matrix defined in the literature considers if a peak was detected in the neighborhood of a reference peak, and has no definition of true negatives '''
        
        # running through the reference peaks one can extract true positives and false negatives
        # false positives = all detected - true positives
        
        # Sampling interval in seconds
        T = 1/self.fs
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
    
    def literature_record_set_confusion_matrix(self, ppg_records):
        ''' Given a set of records containing ppg signals and peak references, returns the confusion (triangular) matrix based on the literature.'''

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            # Detect peaks using current set of parameters
            # _, _, _,_, detected_peaks = self.detect_peaks_cross(ppg_signal)
            _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
            
            detected_locations = self.peak_positions(ppg_signal, detected_peaks)
            # Get record's confusion matrix and regularization term
            tp, fp, fn = self.literature_signal_confusion_matrix(detected_locations, reference_peaks)
            true_positives += tp; false_positives += fp; false_negatives += fn
            
        return true_positives, false_positives, false_negatives
        

    def terma_record_set_confusion_matrix(self, ppg_records):
        ''' Given a set of records containing ppg signals and peak references, returns the TERMA (triangular) confusion matrix based on the literature.'''
    
        import terma
            
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            # Detect peaks using TERMA
            detected_locations = terma.peak_positions_terma(ppg_signal)
            # Get record's confusion matrix and regularization term
            tp, fp, fn = self.literature_signal_confusion_matrix(detected_locations, reference_peaks)
            true_positives += tp; false_positives += fp; false_negatives += fn
            
        return true_positives, false_positives, false_negatives
            
            
# Given the number of iterations and alphas range, performs random search on the crossover's alphas using train data accuracy as fitness metric
def random_search_crossover(train_records, iterations_of_interest, min_alpha, max_alpha, sampling_frequency, verbosity):
    if (min_alpha < 0) or (min_alpha > 1) or (max_alpha < 0) or (max_alpha > 1):
        print('Error, minimum and maximum alphas must be between 0 and 1')
        exit(-1)
    if len(iterations_of_interest) == 0:
        print('Error, iterations of interest must not be empty')
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
        tp, fp, fn = peak_detector.literature_record_set_confusion_matrix(train_records)
        
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
        
#!python3
# Author: Victor O. Costa 

import numpy as np
import sys

class crossover_detector:
    """ Class to process the PPG signal and indicate peaks using a crossover of moving averages """
    
    def __init__(self):
        """ Constructor method """
        ## Parameters
        # Crossover
        self.alpha_fast = 0.3011934482294183 #0.548263
        self.alpha_slow = 0.9482683739254808 #0.964244
        
        # Variance
        self.var_alpha = 0.5
        self.var_threshold = 1.0
        
        ## Variables
        # Crossover
        self.average_fast = 0.0
        self.average_slow = 0.0
        self.crossover_index = 0.0
        self.first_pass = True
        
        # Variance
        self.var_average = 0.0
        self.variance = 0.0
        
        
    def set_parameters(self, alpha_fast, alpha_slow):
        """ Define exponential average and threshold parameters """ 
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        
    def set_parameters_var(self, var_alpha, var_threshold):
        self.var_alpha = var_alpha
        self.var_threshold = var_threshold
        
    def set_parameters_mix(self, alpha_fast, alpha_slow, var_alpha, avg_alpha, var_threshold):
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.var_alpha = var_alpha
        self.avg_alpha = avg_alpha
        self.var_threshold = var_threshold
        
    def exponential_ma(self, alpha, input_value, old_average):
        """ Returns exponential moving average without internal memory """
        return (1-alpha)*input_value + alpha*old_average

    def exponential_mv(self, alpha, input_value, old_variance, old_average):
        # From the paper "Incremental calculation of weighted mean and variance" (Tony Finch, 2009)
        variance = (1 - alpha) * (old_variance + alpha*((input_value - old_average)**2)) 
        return variance
   
    def reset_state(self):
        """ Resets fast and slow averages """
        self.average_fast = 0.0
        self.average_slow = 0.0
        self.first_pass = True
   
    def update_model(self, ppg_value):
        """ Updates current_index in an online way (value by value) """
        # Crossover process
        self.average_fast = self.exponential_ma(self.alpha_fast, ppg_value, self.average_fast)
        self.average_slow = self.exponential_ma(self.alpha_slow, ppg_value, self.average_slow)
        self.crossover_index = self.average_fast - self.average_slow
         
    def update_model_var(self, ppg_value):
        self.variance = self.exponential_mv(self.var_alpha, ppg_value, self.variance, self.var_average)
        self.var_average = self.exponential_ma(1 - self.var_alpha, ppg_value, self.var_average)
        #print("Var: " + str(self.variance) + "\nAvg: " + str(self.variance))
        
    def update_model_mix(self, ppg_value):
        self.average_fast = self.exponential_ma(self.alpha_fast, ppg_value, self.average_fast)
        self.average_slow = self.exponential_ma(self.alpha_slow, ppg_value, self.average_slow)
        self.crossover_index = self.average_fast - self.average_slow
        
        self.variance = self.exponential_mv(self.var_alpha, self.crossover_index, self.variance, self.var_average)
        self.var_average = self.exponential_ma(self.avg_alpha, self.crosssover_index, self.var_average)
        
    # Crossover based peak detection
    def detect_peaks(self, ppg_array):
        self.reset_state()
        fast_averages = []
        slow_averages = []
        crossover_indices = []
        peaks_array = []
                
        for value in ppg_array:
            self.update_model(value)
            
            fast_averages.append(self.average_fast)
            slow_averages.append(self.average_slow)
            crossover_indices.append(self.crossover_index)
                       
            # Crossover detection
            if self.crossover_index > 0:
            #if self.crossover_index > 0:
                peaks_array.append(1)
            else:
                peaks_array.append(0)
        
        return fast_averages, slow_averages, crossover_indices, peaks_array
        
        
    def detect_peaks_var(self, ppg_array):
        self.reset_state()
        averages = []
        variances = []
        peaks_array = []
                
        for value in ppg_array:
            self.update_model_var(value)
            variances.append(self.variance)
            averages.append(self.var_average)
            
            if value > self.var_average + self.var_threshold * self.variance:
                peaks_array.append(1)
            else:
                peaks_array.append(0)
            
        return variances, averages, peaks_array    
        
    def detect_peaks_mix(self, ppg_array):
        self.reset_state()
        # fast_averages = []
        # slow_averages = []
        # crossover_indices = []
        # averages = []
        # variances = []
        peaks_array = []
        
        for value in ppg_array:
            self.update_model_var(value)
            if self.crossover_index > self.var_average + self.var_threshold * self.variance:
                peaks_array.append(1)
            else:
                peaks_array.append(0)
                
        return peaks_array
        
        
    def calculate_heart_rates(self, peaks_array, freq):
        """ Use the beats to calculate Heart Rates in bpm """
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
        
    def signal_confusion_matrix(self, detected_peaks, peaks_reference):
        """ Given a set of detected peaks and peaks reference, returns the confusion matrix."""
        # DEPRECATED DESCRIPTION
        # Confusion matrix:
        # Between a falling and a rising edge: for 0 reference peaks, TN++;    for N reference peaks, FN += N
        # Between a rising and a falling edge: for N reference peaks, TP += 1; for 0 reference peaks, FP++
        # In this way, TP + FN = len(peaks_reference)
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
                        #print("True negative at index = ", index) 
                        true_negatives += 1
                    # For one or more reference peaks in a prediction valley, consider the false negatives and true negatives around it
                    else:
                        confusion_array.append(('fn',index))
                        #print("False negative at index = ", index) 
                        false_negatives += state_peaks
                        true_negatives += state_peaks + 1
                
                # Falling edge
                elif state == 1:
                    # For no reference peaks in a prediction hill, increments false positives.
                    if state_peaks == 0:
                        confusion_array.append(('fp',index))
                        #print("False positive at index = ", index) 
                        if not fp_hill_flag:
                            false_positives += 1
                        # If the false positive is preceded by a true negative, it means that the previous and next true negatives must be ignored
                        true_negatives -= 1
                        fp_hill_flag = True
                    # For more than one reference peaks in a prediction hill, increments the true positives and the false positives with reference to the reference valleys between ref. peaks 
                    else:
                        confusion_array.append(('tp',index))
                        #print("True positive at index = ", index) 
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
            

    def record_confusion_matrix(self, ppg_records):
        """ Given a set of records containing ppg signals and peak references, returns the confusion matrix."""
        
        true_positives = 0 
        true_negatives = 0 
        false_positives = 0
        false_negatives = 0
        
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Detect peaks using current set of parameters
            _, _, _, detected_peaks = self.detect_peaks(ppg_signal)
            
            # Get record's confusion matrix and regularization term
            tp, tn, fp, fn, _ = self.signal_confusion_matrix(detected_peaks, reference_peaks)
            true_positives += tp; true_negatives += tn; false_positives += fp; false_negatives += fn
                        
        return true_positives, true_negatives, false_positives, false_negatives
        
            
    def signal_regularization(self, detected_peaks, peaks_reference):
        """ Given a set of detected peaks and peaks reference, returns the regularization value, considering total prediction area and number of predicted peaks. """        
        # Assumes that detected_peaks treats 0 as negative prediction and 1 as positive prediction
        # Considers the proportion between positively predicted and total prediction area to make the positive prediction area small
        positive_predictions = sum(detected_peaks)
        total_predictions = len(detected_peaks)
        area_term = positive_predictions/total_predictions                                                                          # Maximum area_term value is 1
        #print('Area term = ', area_term)
        
        # Considers the number of detected peaks and reference peaks to make them close
        # detected_peaks = np.array(detected_peaks)
        # rising_edges_mask = np.flatnonzero((detected_peaks[:-1] == 0) & (detected_peaks[1:] == 1))                                  # Mask to get the rising edges indices
        # predicted_peaks_number = len(rising_edges_mask)
        # reference_peaks_number = len(peaks_reference)
        # number_term = abs(reference_peaks_number - predicted_peaks_number)/max(reference_peaks_number, predicted_peaks_number)      # Maximum number_term value is 1 
        # Total regularization is the average between area and number terms, having a maximum value of 1
        #total_regularization = (area_term + number_term)/2
        
        return area_term
        
    def total_regularized_cost(self, ppg_records, C, method):
        """ Given a set of PPG records cont and the correspondent peak references, calculates a confusion matrix-based metric, regularized by the total area and number of peaks detected.  """
        total_cost = 0.0
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Detect peaks using current set of parameters
            if method == 'crossover':
                _, _, _, detected_peaks = self.detect_peaks(ppg_signal)
            elif method == 'variance':
                _, _, detected_peaks = self.detect_peaks_var(ppg_signal)
            elif method == 'mix':
                detected_peaks = self.detect_peaks_mix(ppg_signal)
            else:
                print("No match for method argument in total cost")
                sys.exit(-1)
                
            # Get record's confusion matrix and regularization term
            tp, tn, fp, fn, _ = self.signal_confusion_matrix(detected_peaks, reference_peaks)

            #print('[RECORD ', index,']')
            print('TP: ', tp, 'TN: ', tn, 'FP: ', fp, 'FN: ', fn)
            #print('Number of reference peaks: ', len(reference_peaks))
            
            record_regularization = self.signal_regularization(detected_peaks, reference_peaks)
            
            # Calculate record's accuracy and cost
            record_accuracy = (tp + tn)/(tp + tn + fp + fn)
            record_cost = (1 - record_accuracy) + C * record_regularization
            total_cost += record_cost
            
        total_cost /= len(ppg_records)
        
        return total_cost
        
    
    # Given a solution archive and a record set, returned the detected peaks by the rule of majority voting
    def crossover_bagging_detection(self, alphas_fast, alphas_slow, record_set):
            if len(alphas_fast) != len(alphas_slow):
                print("Alphas lengths do not match")
                exit(-1)
            
            individual_predictions = []
            for i in range(0, len(alphas_fast)):
                _, _, _, detected_peaks = self.detect_peaks(ppg_signal)
                individual_predictions.append(detect_peaks)
            
            voted_peaks = (np.sum( individual_predictions, axis=0) > float(ensemble_size)/2) * 1
        
        return voted_peaks
    
    
    
    
    
    
    
    
    
    
    
    
    
    
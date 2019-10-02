#!python3
# Author: Victor O. Costa 

import numpy as np

class crossover_detector:
    """ Class to process the PPG signal and indicate peaks using a crossover of moving averages """
    
    def __init__(self):
        """ Constructor method """
        self.alpha_fast = 0.3011934482294183 #0.548263
        self.alpha_slow = 0.9482683739254808 #0.964244
        self.average_fast = 0.0
        self.average_slow = 0.0
        self.crossover_index = 0.0
    
    def set_parameters(self, alpha_fast, alpha_slow):
        """ Define exponential average and threshold parameters """ 
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        
    def exponential_ma(self, alpha, input_value, old_average):
        """ Returns exponential moving average without internal memory """
        return (1-alpha)*input_value + alpha*old_average

    def kaufman_ma():
        """ Returns Kaufman's adaptative moving average without internal memory """
        pass
   
    def reset_averages(self):
        """ Resets fast and slow current values """
        self.average_fast = 0.0
        self.average_slow = 0.0
   
    def update_averages(self, ppg_value):
        """ Updates current_index in an online way (value by value) """
        self.average_fast = self.exponential_ma(self.alpha_fast, ppg_value, self.average_fast)
        self.average_slow = self.exponential_ma(self.alpha_slow, ppg_value, self.average_slow)
        self.crossover_index = self.average_fast - self.average_slow
   
    def detect_peaks(self, ppg_array):
        self.reset_averages()
        fast_averages = []
        slow_averages = []
        crossover_indices = []
        peaks_array = []
        
        for value in ppg_array:
            self.update_averages(value)
            fast_averages.append(self.average_fast)
            slow_averages.append(self.average_slow)
            crossover_indices.append(self.crossover_index)
            if self.crossover_index > 0:
                peaks_array.append(1)
            else:
                peaks_array.append(0)
        
        return fast_averages, slow_averages, crossover_indices, peaks_array
        
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
        state = 0                                                       # Initial state is negative prediction (no peak)
        state_peaks = 0                                                 # Number of peaks for a given state 
        ref_index = 0                                                   # Index of the reference peaks array
        tn_flag = False                                                 # Flag to avoid a falsely detected peak in one valley to generate two true negatives for that valley 
        for index, prediction in enumerate(detected_peaks):
            # Updates confusion matrix when reaches an edge 
            if prediction != state:
                # Rising edge
                if state == 0:
                    # For no reference peaks in a prediction valley, increment true negatives by one
                    if state_peaks == 0:
                        #print("True negative at index = ", index) 
                        true_negatives += 1
                    # For one or more reference peaks in a prediction valley, consider the false negatives and true negatives around it
                    else:
                        #print("False negative at index = ", index) 
                        false_negatives += state_peaks
                        true_negatives += state_peaks + 1
                
                # Falling edge
                elif state == 1:
                    # For no reference peaks in a prediction hill, increments false positives.
                    if state_peaks == 0:
                        #print("False positive at index = ", index) 
                        false_positives += 1
                        # If the false positive is preceded by a true negative, it means that the previous and next true negatives must be ignored
                        true_negatives -= 1
                        fp_hill_flag = True
                    # For more than one reference peaks in a prediction hill, increments the true positives and the false positives with reference to the reference valleys between ref. peaks 
                    else:
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
        detected_peaks = np.array(detected_peaks)
        rising_edges_mask = np.flatnonzero((detected_peaks[:-1] == 0) & (detected_peaks[1:] == 1))                                  # Mask to get the rising edges indices
        predicted_peaks_number = len(rising_edges_mask)
        reference_peaks_number = len(peaks_reference)
        number_term = abs(reference_peaks_number - predicted_peaks_number)/max(reference_peaks_number, predicted_peaks_number)      # Maximum number_term value is 1 
        
        # Total regularization is the average between area and number terms, having a maximum value of 1
        total_regularization = (area_term + number_term)/2
        
        return area_term
        
    def total_regularized_cost(self, ppg_records, C):
        """ Given a set of PPG records cont and the correspondent peak references, calculates a confusion matrix-based metric, regularized by the total area and number of peaks detected.  """
        total_cost = 0.0
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.hrv[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Detect peaks using current set of parameters
            _, _, _, detected_peaks = self.detect_peaks(ppg_signal)
            
            # Get record's confusion matrix and regularization term
            tp, tn, fp, fn = self.signal_confusion_matrix(detected_peaks, reference_peaks)

            #if((tp <= 0) or (tn <= 0) or (fp <= 0) or (fn <= 0)):
            print('[RECORD ', index,']')
            print('TP: ', tp, 'TN: ', tn, 'FP: ', fp, 'FN: ', fn)
            print('Number of reference peaks: ', len(reference_peaks))
            
            #/if

            record_regularization = self.signal_regularization(detected_peaks, reference_peaks)
            
            # Calculate record's accuracy and cost
            record_accuracy = (tp + tn)/(tp + tn + fp + fn)
            record_cost = (1 - record_accuracy) + C * record_regularization
            total_cost += record_cost
            
        total_cost /= len(ppg_records)
        
        return total_cost
        
    # DEPRECATED RMSE OF FEATURES BY MINUTE
    # def _group_signal(signal, fs, time):
        # """ Groups a given signal in n items at a time, being n dependent on the sampling frequency (Hz) and wanted time (s) """
        # n = fs * time
        # iterable_groups = zip(*([iter(signal)]*n))
        # return iterable_groups
        
    # def _group_hr_reference(heart_rates, minute):
        # """ Groups reference hrv in a list according to the wanted minute (eg. minute = 0, returns hrv from 00:00 to 01:00) """
        # sum = 0.0
        # grouped_hr = []
        # for value in heart_rates:
            # sum += value
            # if sum > 60*minute and sum < 60*(minute+1):
                # grouped_hr.append(value)
                
        # return grouped_hr
    
    # def features_rmse(self, ppg_signals, hr_references, fs, features_extractor):
        # """ Returns RMSE of the estimated hrv parameters by minute on each subject using given hrv references """
        # total_error = 0.0
        # for subject, subject_signal in enumerate(ppg_signals):
            # subject_hr = hr_references[subject]
            # subject_error = 0.0
            # #FIX: iterating in groups
            # for current_minute, signal_minute in enumerate(self._group_signal(subject_signal, fs, 60)):            # loops signal in groups of 60 seconds
                # # Estimate HRV using current alphas and calculate parameters
                # _, _, _, detected_peaks = self.detect_peaks(signal_minute)
                # detected_heart_rates = calculate_heart_rates(detect_peaks, fs)
                # hr_features = features_extractor(detected_heart_rates)
                # # Get reference HRV for the current minute and calculate its parameters
                # minute_ref_hr = _group_hr_reference(subject_hr, minute=current_minute)
                # reference_features = features_extractor(minute_ref_hr)
                # # Calculate RMSE in parameters space
                # minute_error = ((np.array(hr_features) - np.array(reference_features))**2)/len(hr_features)           # normalize minute error by number of parameters
                # subject_error += minute_error  
            # subject_error /= 15                         # normalize subject error by number of minutes
            # total_error += subject_error
        # total_error /= len(ppg_signals)                 # normalize total error by number of subjects
        
        # return total_error
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
        self.alpha_crossover =  0.0 
        self.alpha_fast =       0.0 #0.548263
        self.alpha_slow =       0.0 #0.964244

        
        # Variance
        self.var_alpha = 0.5
        self.var_threshold = 1.0
       
        ## Variables
        # Crossover
        self.average_fast = 0.0
        self.average_slow = 0.0
        self.crossover_index = 0.0
        
        # Variance
        self.var_average = 0.0
        self.variance = 0.0
        
        
    def set_parameters_cross(self, alpha_crossover, alpha_fast, alpha_slow):
        """ Define exponential average and threshold parameters """ 
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.alpha_crossover = alpha_crossover
        
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

        
    def reset_state(self, first_ppg_val):
        """ Resets fast and slow averages """
        self.average_fast = first_ppg_val
        self.average_slow = first_ppg_val
        self.crossover_index = 0.0
        
    def update_model(self, ppg_value):
        """ Updates current_index in an online way (value by value) """
        # Crossover process
        self.average_fast       = self.exponential_ma(self.alpha_fast       , ppg_value, self.average_fast)
        self.average_slow       = self.exponential_ma(self.alpha_slow       , ppg_value, self.average_slow)
        self.crossover_index    = self.exponential_ma(self.alpha_crossover  , self.average_fast - self.average_slow, self.crossover_index)
        
        
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
        
        
    def detect_peaks_cross(self, ppg_array):
        """ Crossover based peak detection  """
        self.reset_state(ppg_array[0])
        
        fast_averages = []
        slow_averages = []
        crossover_indices = []
        peaks_array = []
        
        
        from scipy.signal import butter, lfilter, lfilter_zi
        # This function will apply the filter considering the initial transient.
        
        def butter_lowpass(highcut, sRate, order):
            nyq = 0.5 * sRate
            high = highcut / nyq
            b, a = butter(order, high, btype='low')
            return b, a
        
        def butter_bandpass_filter_zi(data, highcut, sRate, order):
            b, a = butter_lowpass(highcut, sRate, order)
            zi = lfilter_zi(b, a)
            y, zo = lfilter(b, a, data, zi=zi*data[0])
            return y
        
        sRate = 200
        highcut = 8
        order = 2
        ppg_array = butter_bandpass_filter_zi(ppg_array, highcut, sRate, order)
        
        for index, value in enumerate(ppg_array):
            self.update_model(value)

            fast_averages.append(self.average_fast)
            slow_averages.append(self.average_slow)
            crossover_indices.append(self.crossover_index)

            if self.crossover_index > 0:
                peaks_array.append(1)
            else:  
                peaks_array.append(0)
            
        
        return fast_averages, slow_averages, crossover_indices, peaks_array
        
        
    def detect_peaks_var(self, ppg_array):
        self.reset_state(ppg_array[0])
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
        self.reset_state(ppg_array[0])
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
        
    
    def combine_peak_predictions(self, record_predictions, models_weights, threshold):
        """ Given a matrix containing the predictions of all ensemble models for a specific record (num_models x signal_len), return the weighted voting result prediction over the record's peaks """
        if len(record_predictions) != len(models_weights):
            print("Number of models in predictions and in weights must be the same")
            exit(-1)
        
        ensemble_size = len(record_predictions)
        
        # Weight predictions matrix according to each model's weights
        weighted_predictions_matrix = []
        for index, model_predictions in enumerate(record_predictions):
            weighted_predictions_matrix.append(list( np.array(model_predictions) * models_weights[index] ))
            
        ensemble_predictions = ( (np.sum(weighted_predictions_matrix, axis=0) / float(ensemble_size)) > threshold ) * 1
        
        return ensemble_predictions
      
      
    def ignore_short_peaks(self, detected_peaks, peak_len_threshold):
        """ Ignore peaks with low duration  """
        peaks_array = list(detected_peaks)
        peak_len_counter = 0
        for index, peak_state in enumerate(peaks_array):
            if peak_state == 1:
                peak_len_counter += 1
            else:
                if peak_len_counter < peak_len_threshold and peak_len_counter > 0:
                    peaks_array[(index - peak_len_counter) : index] = [0]*peak_len_counter
                peak_len_counter = 0    
        
        return peaks_array
        
    def peak_positions(self, ppg_signal, detected_peaks):
        """ From detected peak regions, extract exact peak locations """
        if len(ppg_signal) != len(detected_peaks):
            print("PPG signal and detected peaks do not match")
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
                    
            # Find peak locations when "detected_peaks" is high (1-valued)
            if flag_max and ppg_signal[i] > current_peak:
                current_peak = ppg_signal[i]
                current_location = i
        
        # Include possible last peak not detected because falling edge did not occurred
        if flag_max:
            peak_positions.append(current_location)
        
        return peak_positions
        
        
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
    

    def record_set_confusion_matrix(self, ppg_records, method, large_peaks_only, peak_len_threshold):
        """ Given a set of records containing ppg signals and peak references, returns the confusion matrix."""
        
        if method != 'crossover' and method != 'variance' and method != 'mix':
            print("Please choose a valid method")
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
            if method == 'crossover':
                # _, _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
                _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
            elif method == 'variance':
                _, _, detected_peaks = self.detect_peaks_var(ppg_signal)
            elif method == 'mix':
                detected_peaks = self.detect_peaks_mix(ppg_signal)
            else:
                print("No match for method argument in total cost")
                sys.exit(-1)
            if large_peaks_only == True:
                detected_peaks = self.ignore_short_peaks(detected_peaks, peak_len_threshold)
            
            
            # Get record's confusion matrix and regularization term
            tp, tn, fp, fn, _ = self.signal_confusion_matrix(detected_peaks, reference_peaks)
            true_positives += tp; true_negatives += tn; false_positives += fp; false_negatives += fn
                        
        return true_positives, true_negatives, false_positives, false_negatives
    
    
    def literature_signal_confusion_matrix(self, detected_locations, reference_locations):
        """ The confusion (triangular) matrix defined in the literature considers if a peak was detected in the neighborhood of a reference peak, and has no definition of true negatives """
        
        # running through the reference peaks one can extract true positives and false negatives
        # false positives = all detected - true positives
        
        # Sampling interval in seconds
        T = 0.005
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
    
    
    def literature_record_set_confusion_matrix(self, ppg_records, large_peaks_only, peak_len_threshold):
        """ Given a set of records containing ppg signals and peak references, returns the confusion (triangular) matrix based on the literature."""

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
            
            if large_peaks_only == True:
                detected_peaks = self.ignore_short_peaks(detected_peaks, peak_len_threshold)
            
            detected_locations = self.peak_positions(ppg_signal, detected_peaks)
            # Get record's confusion matrix and regularization term
            tp, fp, fn = self.literature_signal_confusion_matrix(detected_locations, reference_peaks)
            true_positives += tp; false_positives += fp; false_negatives += fn
            
        return true_positives, false_positives, false_negatives
        
    def terma_record_set_confusion_matrix(self, ppg_records):
        """ Given a set of records containing ppg signals and peak references, returns the TERMA (triangular) confusion matrix based on the literature."""
    
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
        
    
    def ensemble_records_confusion_matrix(self, records, records_predictions, models_weights, decision_threshold, large_peaks_only, peak_len_threshold):
        """ Given the predictions of each ensemble's model over a set of records (num_records x num_models x record_len), a set of model weights and a threshold, return the weighted voting confusion matrix """
        if len(records) != len(records_predictions):
            print("Number of records do not match")
            exit(-1)
    
        if len(records_predictions[0]) != len(models_weights):
            print("Number of models and weights are different")
            exit(-1)
            
        true_positives = 0 
        true_negatives = 0 
        false_positives = 0
        false_negatives = 0
        
        # For each record, compute the combined predictions of all models and extract the resulting confusion matrix
        for index in range(len(records)):
            single_record = records[index]
            single_record_predictions = records_predictions[index]
            
            # Combine all models' predictions over the given record
            combined_predictions = self.combine_peak_predictions(single_record_predictions, models_weights, decision_threshold)
            if large_peaks_only == True:
                combined_predictions = self.ignore_short_peaks(combined_predictions, peak_len_threshold)
            # Extract reference peaks for comparison
            reference_peaks = np.array(single_record.beats[0]) - single_record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Get record's confusion matrix
            tp, tn, fp, fn, _ = self.signal_confusion_matrix(combined_predictions, reference_peaks)
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
        """ Given a set of PPG records and the correspondent peak references, calculates a regularized confusion matrix-based metric """
        total_cost = 0.0
        for index, record in enumerate(ppg_records):
            #print('Cost calculation for record ', index)
            ppg_signal = record.ppg[1]
            reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
            
            # Detect peaks using current set of parameters
            if method == 'crossover':
                # _, _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
                _, _, _, detected_peaks = self.detect_peaks_cross(ppg_signal)
            #elif method == 'ensemble':
            #    detected_peaks = self.ensemble_join_detections(models, weights, ppg_signal)
            elif method == 'variance':
                _, _, detected_peaks = self.detect_peaks_var(ppg_signal)
            elif method == 'mix':
                detected_peaks = self.detect_peaks_mix(ppg_signal)
            else:
                print("No match for method argument in total cost")
                sys.exit(-1)
                
            # Get record's confusion matrix and regularization term
            tp, tn, fp, fn, _ = self.signal_confusion_matrix(detected_peaks, reference_peaks)
            true_positive_rate = tp / (tp + fn)
            true_negative_rate = tn / (tn + fp)
            #print('[RECORD ', index,']')
            #print('TP: ', tp, 'TN: ', tn, 'FP: ', fp, 'FN: ', fn)
            #print('Number of reference peaks: ', len(reference_peaks))
            
            #record_regularization = self.signal_regularization(detected_peaks, reference_peaks)
            
            # Calculate record's accuracy and cost
                     
            record_accuracy = (tp + tn)/(tp + tn + fp + fn)
            record_cost = (1 - record_accuracy) #+ C * record_regularization
            #record_cost = true_positive_rate + C * true_negative_rate
            total_cost += record_cost

        total_cost /= len(ppg_records)
        
        return total_cost
        
        
    
# Given the number of iterations and alphas range, performs random search on the crossover's alphas using train data accuracy as fitness metric
#def random_search_crossover(train_records, num_iterations, min_alpha, max_alpha, min_threshold, max_threshold, large_peaks_only, verbosity):
def random_search_crossover(train_records, num_iterations, min_alpha, max_alpha, large_peaks_only, verbosity):
    if (min_alpha < 0) or (min_alpha > 1) or (max_alpha < 0) or (max_alpha > 1):
        print("Minimum and maximum alphas must be between 0 and 1")
        exit(-1)
    
    if verbosity != False and verbosity != True:
        print("Verbosity must be boolean")
        exit(-1)
    
    peak_detector = crossover_detector()
    best_solution = [0, 0, float('inf')]
    
    # Optimization loop
    for iteration in range(num_iterations):
        if verbosity == True: print('\n[Search iteration ' + str(iteration) + ']')

        # Randomize alphas, with fast alpha depending on slow alpha, thus guaranteeing fast alpha < slow alpha
        alpha_fast = np.random.uniform(min_alpha, max_alpha)
        alpha_slow = np.random.uniform(alpha_fast, max_alpha)
        alpha_crossover = np.random.uniform(min_alpha, max_alpha)
        peak_len_threshold = np.random.randint(0, 20)
        peak_detector.set_parameters_cross(alpha_crossover, alpha_fast, alpha_slow)

        # Run the detector defined above in the train records and extract accuracy
        # tp, tn, fp, fn = peak_detector.record_set_confusion_matrix(train_records, "crossover", large_peaks_only, peak_len_threshold)
        # accuracy = float(tp + tn) / float(tp + tn + fp + fn)
        # cost = 1 - accuracy
        
        # Run the detector defined above in the train records and extract SE and P+
        tp, fp, fn = peak_detector.literature_record_set_confusion_matrix(train_records, large_peaks_only, peak_len_threshold)
        
        SE = tp / (tp + fn)
        Pp = tp / (tp + fp)
        cost = 1 - (SE + Pp)/2
        
        if cost < best_solution[-1]:
            best_solution = [alpha_crossover, alpha_fast, alpha_slow, peak_len_threshold, cost]
        
        if verbosity == True:
            print('Alpha crossover     Alpha fast     Alpha slow     peak samples thr     cost')
            print('[randomized]\t', alpha_crossover,'\t', alpha_fast, '\t', alpha_slow,'\t', peak_len_threshold, '\t', cost)
            print('[best til now]\t', best_solution[0], '\t', best_solution[1], '\t',  best_solution[2], '\t', best_solution[3],'\t', best_solution[-1])
    
    return best_solution 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
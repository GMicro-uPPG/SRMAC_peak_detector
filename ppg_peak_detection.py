#!python3
# Author: Victor O. Costa 

import numpy as np

class crossover_detector:
    """ Class to process the PPG signal and indicate peaks using a crossover of moving averages """
    
    def __init__(self):
        """ Constructor method """
        self.alpha_fast = 0.05
        self.alpha_slow = 0.98
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
        
    def _group_signal(signal, fs, time):
        """ Groups a given signal in n items at a time, being n dependent on the sampling frequency (Hz) and wanted time (s) """
        n = fs * time
        iterable_groups = zip(*([iter(signal)]*n))
        return iterable_groups
        
    def _group_hr_reference(heart_rates, minute):
        """ Groups reference hrv in a list according to the wanted minute (eg. minute = 0, returns hrv from 00:00 to 01:00) """
        sum = 0.0
        grouped_hr = []
        for value in heart_rates:
            sum += value
            if sum > 60*minute and sum < 60*(minute+1):
                grouped_hr.append(value)
                
        return grouped_hr
        
    def features_rmse(self, ppg_signals, hr_references, fs, features_extractor):
        """ Returns RMSE of the estimated hrv parameters by minute on each subject using given hrv references """
        total_error = 0.0
        for subject, subject_signal in enumerate(ppg_signals):
            subject_hr = hr_references[subject]
            subject_error = 0.0
            #FIX: iterating in groups
            for current_minute, signal_minute in enumerate(self._group_signal(subject_signal, fs, 60)):            # loops signal in groups of 60 seconds
                # Estimate HRV using current alphas and calculate parameters
                _, _, _, detected_peaks = self.detect_peaks(signal_minute)
                detected_heart_rates = calculate_heart_rates(detect_peaks, fs)
                hr_features = features_extractor(detected_heart_rates)
                # Get reference HRV for the current minute and calculate its parameters
                minute_ref_hr = _group_hr_reference(subject_hr, minute=current_minute)
                reference_features = features_extractor(minute_ref_hr)
                # Calculate RMSE in parameters space
                minute_error = ((np.array(hr_features) - np.array(reference_features))**2)/len(hr_features)           # normalize minute error by number of parameters
                subject_error += minute_error  
            subject_error /= 15                         # normalize subject error by number of minutes
            total_error += subject_error
        total_error /= len(ppg_signals)                 # normalize total error by number of subjects
        
        return total_error
     
    # DEPRECATED
    # def parameters_random_search(self, num_iterations, ppg_signals, hrv_references, fs, parameters_calculator):
        # """ Performs random search of the alphas, optimizing RMSE of the estimated hrv parameters by minute on each subject using given hrv references """
      
        # searched_parameters = []
        # for iteration in range(num_iterations):
            # # Randomize alphas 
            # iter_alpha_fast = np.random.uniform(0, 1)
            # iter_alpha_slow = np.random.uniform(0, 1)
            # self.set_parameters(iter_alpha_fast, iter_alpha_slow)
            # iteration_error = 0.0
            
            # for subject, subject_signal in enumerate(ppg_signals):
                # subject_hrv = hrv_references[subject]
                # subject_error = 0.0
                
                # #FIX: iterating in groups
                # for current_minute, signal_minute in enumerate(_group(subject_signal, fs, 60)):            # loops signal in groups of 60 seconds
                    # # Estimate HRV using current alphas and calculate parameters
                    # _, _, _, detected_peaks = self.detect_peaks(signal_minute)
                    # detected_heart_rates = self.calculate_heart_rates(detect_peaks, fs)
                    # detected_parameters = parameters_calculator(detected_heart_rates)
                    # # Get reference HRV for the current minute and calculate its parameters
                    # reference_hrv = self._group_hrv_reference(subject_hrv, minute=current_minute)
                    # reference_parameters = parameters_calculator(reference_hrv)
                    # # Calculate RMSE in parameters space
                    # minute_error = ((np.array(detected_parameters) - np.array(reference_parameters))**2)/len(detected_parameters)
                    # subject_error += minute_error
                    
                # subject_error /= 15                         # normalize subject error by number of minutes
                # iteration_error += subject_error
            
            # iteration_error /= len(ppg_signals)             # normalize iteration error by number of subjects 
            # searched_parameters.append([iter_alpha_fast, iter_alpha_slow, iteration_error])
        
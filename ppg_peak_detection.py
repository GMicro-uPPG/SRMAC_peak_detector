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
              
    def get_peak_blocks(self, raw_ppg):
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
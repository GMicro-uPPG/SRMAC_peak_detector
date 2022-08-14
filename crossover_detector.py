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

# Python standard lib
from collections.abc import Iterable
# Application modules
from base_detector import *
import utilities

class crossover_detector(base_detector):
    ''' Class to process the PPG signal and indicate peaks using a crossover of moving averages '''
    
    def __init__(self, alpha_crossover:float, alpha_fast:float, alpha_slow:float):
        ''' Constructor '''
        
        # Sanity check
        if any([(alpha >= 1) or (alpha <= 0) for alpha in [alpha_crossover, alpha_fast, alpha_slow]]):
            print('Error, alphas must be in 0 <= a <= 1')
            exit(-1)
            
        # Smoothing constants
        self.alpha_crossover =  alpha_crossover 
        self.alpha_fast =       alpha_fast
        self.alpha_slow =       alpha_slow
        
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
              
    def get_peak_results(self, raw_ppg, sampling_frequency):
        ''' Crossover based peak detection  '''
        # Sanity checks
        if not isinstance(raw_ppg, Iterable):
            print('Error, PPG signal must be an iterable')
            exit(-1)
        if len(raw_ppg) == 0:
            print('Error, length of PPG signal must be greater than zero')
            exit(-1)
        if sampling_frequency < 0:
            print('Sampling frequency must be positive')
            exit(-1)
        
        # Reset both EWMA to zero
        self.reset_state(0)
        
        # Dynamics' history
        fast_averages = []
        slow_averages = []
        crossover_indices = []
        peak_blocks = []
        peak_positions = []
        
        # Low-pass filter parameters
        order = 2
        low_cut = 0.5   # Hz
        high_cut = 8    # Hz 
        filtered_ppg = utilities.biquad_butter_bandpass(raw_ppg, order, low_cut, high_cut, sampling_frequency)

        # Current state of peak detection
        fsm_state = STATE_SEEKING_PEAK
        peak_height = float('-inf')
        peak_position = 0
        
        # 
        for index, ppg_sample in enumerate(filtered_ppg):
            # Update model
            self.average_fast       = self.exponential_ma(self.alpha_fast       , ppg_sample, self.average_fast)
            self.average_slow       = self.exponential_ma(self.alpha_slow       , ppg_sample, self.average_slow)
            self.crossover_index    = self.exponential_ma(self.alpha_crossover  , self.average_fast - self.average_slow, self.crossover_index)
            
            fast_averages.append(self.average_fast)
            slow_averages.append(self.average_slow)
            crossover_indices.append(self.crossover_index)
            
            # The peak condition is evaluated and stored in both states
            peak_condition = self.crossover_index > 0
            peak_blocks.append(int(peak_condition))
            
						# STATE SEEKING PEAK
            # In this state, no peak was detected and we wait for a new peak to begin
            if fsm_state == STATE_SEEKING_PEAK:
              # State transition
              if peak_condition:
                fsm_state = STATE_PEAK_FOUND
              
            # STATE PEAK FOUND
						## This state characterizes the current peak and stores its info
            else:
              # Find sample with highest magnitude
              if ppg_sample > peak_height:
                peak_height = ppg_sample
                peak_position = index
              
              # State transition
              ## If the signal ends during a peak block onset, act as if a falling edge occurs
              if (not peak_condition) or (index == len(filtered_ppg) - 1):
                peak_positions.append(peak_position)
                peak_height = float('-inf')
                fsm_state = STATE_SEEKING_PEAK
            
        return  fast_averages, slow_averages, crossover_indices, peak_blocks, peak_positions
        
    # Wrapper to keep conformity with the base_detector and TERMA_detector classes
    def detect(self, raw_ppg, sampling_frequency):
        _, _, _, peak_blocks, peak_locations = self.get_peak_results(raw_ppg, sampling_frequency)
        return peak_blocks, peak_locations
        
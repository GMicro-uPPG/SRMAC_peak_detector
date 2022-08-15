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

# Authors: Cesar Abascal and Victor O. Costa

# Python standard lib
from collections.abc import Iterable
import math
# Third party
import numpy as np
# Application modules
import utilities
from base_detector import *

class TERMA_detector(base_detector):
    ''' Implementation of the peak detector proposed by Elgendi '''
    
    def __init__(self, peak_window_ms:int, beat_window_ms:int, beta:float):
        ''' Constructor '''
        # Sanity check
        if beta < 0.0:
            print('Error, beta must be greater than or equal to 0')
            exit(-1)
        if peak_window_ms <= 0 or beat_window_ms <= 0:
            print('Error, window sizes should be greater than 0')
            exit(-1)

        # Algorithm parameters
        ## W1 - peak window size in milliseconds
        self.peak_window_ms = peak_window_ms 
        ## W2 - beat window size in milliseconds
        self.beat_window_ms = beat_window_ms
        self.beta = beta
    
    def nearest_odd(self, x):
        return 2 * math.floor( x / 2 ) + 1
    
    def simple_moving_average(self, signal, window_len):
        '''  '''
        moving_avg = []
        for index in range(len(signal)-window_len):
            # Compute mean in the range signal[index, index+window-1]
            local_avg = np.mean(signal[index : (index+window_len)])
            moving_avg.append(local_avg)
            
        return moving_avg

    def get_peak_results(self, raw_ppg, sampling_frequency):
        '''  '''
        # Sanity chek
        if not isinstance(raw_ppg, Iterable):
            print('Error, PPG signal must be an iterable')
            exit(-1)
        if len(raw_ppg) == 0:
            print('Error, length of PPG signal must be greater than zero')
            exit(-1)
        if sampling_frequency <= 0.0:
            print('Error, sampling frequency must be greater than 0')
        
        # Conversion of window sizes from miliseconds to number of samples
        peak_window_len = self.nearest_odd(sampling_frequency * (self.peak_window_ms / 1000))
        beat_window_len = self.nearest_odd(sampling_frequency * (self.beat_window_ms / 1000))
        
        # Filter PPG signal
        order = 2
        low_cut_hz = 0.5
        high_cut_hz = 8
        ppg_filtered = utilities.butter_bandpass_2order_0phase(raw_ppg, low_cut_hz, high_cut_hz, sampling_frequency)
        
        # Clip signal
        ppg_signal = ppg_filtered.clip(min = 0)
        
        # Square signal
        ppg_signal = ppg_signal ** 2
        
        # Zero-padding to avoid a false negative in the last peak of a signal
        ## TODO: Make zero-padding optional
        ppg_signal = np.append(ppg_signal, [0] * beat_window_len)        

        # Compute offset of threshold 1
        alpha = self.beta * np.mean(ppg_signal)
        
        # Compute moving averages
        SMA_peak = self.simple_moving_average(ppg_signal, peak_window_len)
        SMA_beat = self.simple_moving_average(ppg_signal, beat_window_len)

        # Since differences between W1 and W2 lead to different lengths
        #   for SMA_beat and SMA_peak, we make both of them to have the same length
        if len(SMA_beat) > len(SMA_peak):
            SMA_beat = SMA_beat[: len(SMA_peak)]
        elif len(SMA_peak) > len(SMA_beat):
            SMA_peak = SMA_peak[: len(SMA_beat)]
        
        # Compute blocks of interest through SMA crossover, along with peak positions
        peak_blocks = []
        peak_positions = []
        block_width = 0
        index = 0
        
        # Current state of peak detection
        fsm_state = STATE_SEEKING_PEAK
        peak_height = float('-inf')
        peak_position = 0
        
        for sample_peak, sample_beat in zip(SMA_peak, SMA_beat):
            # THR1 
            peak_condition = sample_peak > sample_beat + alpha       
            peak_blocks.append(int(peak_condition))

            # STATE SEEKING PEAK
            # In this state, no peak was detected and we wait for a new peak to begin
            if fsm_state == STATE_SEEKING_PEAK:
                if peak_condition:
                    block_width += 1
                    fsm_state = STATE_PEAK_FOUND

            # STATE PEAK FOUND
            ## This state characterizes the current peak and stores its info
            else:
                block_width += 1
                # Find sample with highest magnitude
                if ppg_filtered[index] > peak_height:
                    peak_height = ppg_filtered[index]
                    peak_position = index

                # State transition
                ## If the signal ends during a peak block onset, act as if a falling edge occurs
                if (not peak_condition) or (index == len(SMA_peak) - 1):
                    # THR2
                    if block_width >= peak_window_len:               
                        peak_positions.append(peak_position)
                    else:
                        peak_blocks[index - block_width + 1: index] = [0] * (block_width - 1)
                        
                    block_width = 0
                    peak_height = float('-inf')
                    fsm_state = STATE_SEEKING_PEAK

            index += 1

        return SMA_peak, SMA_beat, peak_blocks, peak_positions
        
        
    def detect(self, raw_ppg, sampling_frequency):
        _, _, peak_blocks, peak_positions = self.get_peak_results(raw_ppg, sampling_frequency)
        return peak_blocks, peak_positions
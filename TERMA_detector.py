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

# Third party
import numpy as np
# Python standard lib
from collections.abc import Iterable
# Application modules
import utilities
from base_detector import base_detector

class TERMA_detector(base_detector):
    ''' Implementation of the peak detector proposed by Elgendi '''
    def __init__(self, window_peak:int, window_beat:int, beta:float, sampling_frequency:float):
        ''' Constructor '''
        
        # Sanity check
        if beta <= 0.0:
            print('Error, beta must be greater than 0')
            exit(-1)
        if sampling_frequency <= 0.0:
            print('Error, sampling frequency must be greater than 0')
        if window_peak <= 0 or window_beat <= 0:
            print('Error, window sizes should be greater than 0')
            exit(-1)
        
        # Algorithm parameters
        self.W1 = window_peak
        self.W2 = window_beat
        self.beta = beta
        
    def simple_moving_average(self, signal, window):
        '''  '''
        moving_avg = []
        for index in range(len(signal)-window):
            # Compute mean in the range signal[index, index+window-1]
            local_avg = np.mean(signal[index : (index+window)])
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
        
        # Filter signal
        order = 2
        low_cut = 0.5   # Hz
        high_cut = 8    # Hz    
        ppg_filtered = utilities.biquad_butter_bandpass(raw_ppg, order, low_cut, high_cut, sampling_frequency)
        
        # Clip signal
        ppg_signal = ppg_filtered.clip(min = 0)
        
        # Square signal
        ppg_signal = ppg_signal ** 2
        
        # Compute offset of threshold 1
        alpha = self.beta * np.mean(ppg_signal)
        
        # Compute moving averages
        SMA_peak = self.simple_moving_average(ppg_signal, self.W1)
        SMA_beat = self.simple_moving_average(ppg_signal, self.W2)
        
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
        peak_hei = 0.0
        peak_pos = 0
        for sample_peak, sample_beat in zip(SMA_peak, SMA_beat):
            # 
            peak_condition = sample_peak > sample_beat + alpha       # THR1
            peak_blocks.append(int(peak_condition))
            # Updates the height and position of a peak if the condition is True
            if peak_condition:
                block_width += 1
                if ppg_filtered[index] > peak_hei:
                    peak_hei = ppg_filtered[index]
                    peak_pos = index
                # If the signal ends during a peak block onset, act as if a falling edge occurs
                if index == len(SMA_peak) - 1:
                    peak_positions.append(peak_pos)
            # 
            else:
                if block_width >= self.W1:              # THR2    
                    peak_positions.append(peak_pos)
                block_width = 0
                peak_hei = 0.0 
            
            index += 1
            
        return SMA_peak, SMA_beat, peak_blocks, peak_positions
        
        
    def detect(self, raw_ppg, sampling_frequency):
        _, _, peak_blocks, peak_positions = self.get_peak_results(raw_ppg, sampling_frequency)
        return peak_blocks, peak_positions
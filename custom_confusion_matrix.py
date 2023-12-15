#!python3

# MIT License

# Copyright (c) 2023 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

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

def custom_signal_confusion_matrix(peak_blocks, peaks_reference):
    ''' Given a set of detected peaks and peaks reference,
        returns an in-house confusion matrix which has all measures, including true negatives. '''
    # Our confusion matrix, not used in the literature
    
    true_positives = 0 
    true_negatives = 0 
    false_positives = 0
    false_negatives = 0
    
    fp_hill_flag = False
    
    # Assumes that the peak_blocks array treats 0 as negative prediction and 1 as positive prediction
    state_peaks = 0                                                 # Number of peaks for a given state 
    ref_index = 0                                                   # Index of the reference peaks array
    tn_flag = False                                                 # Flag to avoid a falsely detected peak in one valley to generate two true negatives for that valley 
    confusion_array = []                                            # Array to keep which positions corresponds to which CM members
    for index, prediction in enumerate(peak_blocks):
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
    # if index == len(peak_blocks) - 1:
        
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

def custom_record_set_confusion_matrix(peak_detector, ppg_records, sampling_frequency):
    ''' Given a peak detector and a set of records containing ppg signals and peak references,
        returns an in-house confusion matrix which has all measures, including true negatives..'''
    
    true_positives = 0 
    true_negatives = 0 
    false_positives = 0
    false_negatives = 0
    
    for index, record in enumerate(ppg_records):
        #print('Cost calculation for record ', index)
        ppg_signal = record.ppg[1]
        reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
        
        # Detect peaks using current set of parameters
        peak_blocks, _ = peak_detector.detect(ppg_signal, sampling_frequency)
    
    # Get record's confusion matrix and regularization term
    tp, tn, fp, fn, _ = custom_signal_confusion_matrix(peak_blocks, reference_peaks)
    true_positives += tp; true_negatives += tn; false_positives += fp; false_negatives += fn
                
    return true_positives, true_negatives, false_positives, false_negatives

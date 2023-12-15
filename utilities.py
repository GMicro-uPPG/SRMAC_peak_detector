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


def biquad_butter_lowpass(raw_signal, order, cut_frequency, sampling_frequency):
    ''' Filters a signal using a lowpass Butterworth filter, with a biquadratic IIR implementation.
        The initial state of the filter is equal to the first sample of the signal to avoid a large initial overshoot.'''

        # 
    nyquist_freq = 0.5 * sampling_frequency
    normalized_cut = cut_frequency / nyquist_freq
        
    # Define the IIR Buterworth biquad lowpass filter
    sos = scipy.signal.iirfilter(N=order, Wn=normalized_cut, btype='lowpass', analog=False, ftype='butter', output='sos')
    # Defines the initial state and filters signal
        
    zi = scipy.signal.sosfilt_zi(sos) * raw_signal[0]
    filtered_ppg, zo = scipy.signal.sosfilt(sos=sos, x=raw_signal, zi=zi)

    return filtered_ppg
        
        
def biquad_butter_bandpass(raw_signal, order, low_cut, high_cut, sampling_frequency):

    ''' Filters a signal using a bandpass Butterworth filter, with a biquadratic IIR implementation.
        The initial state of the filter is equal to the first sample of the signal to avoid a large initial overshoot.'''

    #
    nyquist_freq = 0.5 * sampling_frequency
    normalized_frequencies = np.array([low_cut, high_cut]) / nyquist_freq
        
    # Define the IIR Buterworth biquad bandpass filter
    sos = scipy.signal.iirfilter(N=order, Wn=normalized_frequencies, btype='bandpass', analog=False, ftype='butter', output='sos')
    
    # Defines the initial state and filters signal
    zi = scipy.signal.sosfilt_zi(sos) * raw_signal[0]
    filtered_ppg, zo = scipy.signal.sosfilt(sos=sos, x=raw_signal, zi=zi)

    return filtered_ppg


def butter_bandpass_2order_0phase(raw_signal, low_cut, high_cut, sampling_frequency):

    ''' Filters a signal forward and backwards with a first-order bandpass Butterworth filter.
        The initial state of the filter is equal to the first sample of the signal to avoid a large initial overshoot.'''

    #
    nyquist_freq = 0.5 * sampling_frequency
    normalized_frequencies = np.array([low_cut, high_cut]) / nyquist_freq

    # Define the FIR Butterworth first-order bandpass filter
    # b, a = scipy.signal.butter(N=1, Wn=normalized_frequencies, btype='bandpass', analog=False, output='ba')
    
    # Define IIR Butterworth first-order bandpass filter
    b, a = scipy.signal.iirfilter(N=1, Wn=normalized_frequencies, btype='bandpass', analog=False, output='ba', ftype='butter')
    
    # Apply filter forward and backwards to achieve a second-order zero-phase filtering
    filtered_ppg = scipy.signal.filtfilt(b, a, raw_signal)
    
    return np.array(filtered_ppg)


def signal_confusion_matrix(peak_locations, reference_locations, sampling_frequency):
    ''' The confusion (triangular) matrix defined in the literature considers if a peak was detected in the
        neighborhood of a reference peak, and has no definition of true negatives '''

    # running through the reference peaks one can extract true positives and false negatives
    # false positives = all detected - true positives

    # Sampling interval in seconds
    T = 1/sampling_frequency
    # Neighborhood definition in seconds
    neighborhood_time = 0.1
    # Neighborhood definition in number of samples
    neighborhood_samples = int(neighborhood_time / T)

    true_positives = 0 
    false_positives = 0
    false_negatives = 0
    i_start = 0

    for reference in reference_locations:
        # Detection of the reference peak is initially assumed to be false
        correct_detection = False

        for i in range(i_start, len(peak_locations)):
            location = peak_locations[i]

            if location > (reference - neighborhood_samples) and location < (reference + neighborhood_samples):
                correct_detection = True
                i_start = i + 1
                break

            if location > reference + neighborhood_samples:
                break

        if correct_detection:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = len(peak_locations) - true_positives

    return true_positives, false_positives, false_negatives    


def record_set_confusion_matrix(peak_detector, ppg_records, sampling_frequency):
    ''' Given a peak detector and a set of records containing ppg signals and peak references,
        returns the confusion (triangular) matrix based on the literature.'''

        # 
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for index, record in enumerate(ppg_records):
        #print('Cost calculation for record ', index)
        ppg_signal = record.ppg[1]
        reference_peaks = np.array(record.beats[0]) - record.ppg[0][0]            # Shifts reference peaks so it is in phase with ppg_signal
        # Detect peaks using the 
        peak_blocks, peak_locations = peak_detector.detect(ppg_signal, sampling_frequency)
        #peak_locations = peak_positions(ppg_signal, peak_blocks)

        # Get record's confusion matrix and regularization term
        tp, fp, fn = signal_confusion_matrix(peak_locations, reference_peaks, sampling_frequency)
        true_positives += tp; false_positives += fp; false_negatives += fn

    return true_positives, false_positives, false_negatives
 
    
def calculate_heart_rates(peaks_array, freq):
    ''' Use the beats to calculate Heart Rates in bpm '''
        
        #
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

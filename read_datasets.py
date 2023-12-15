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

import os
import numpy as np
import time_manager
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi


# Record
class record:
    def __init__(self, name, x_ppg, ppg, x_beats, beats):
        self.name = name # Record name
        self.ppg = [x_ppg, ppg] # Record ppg
        self.beats = [x_beats, beats] # Record beats
    #end-def
#end-class


# # Butterworth highpass filter ------------------------------------------------------------
# def butter_highpass(highcut, sRate, order=5):
    # nyq = 0.5 * sRate
    # high = highcut / nyq
    # b, a = butter(order, high, btype='high')
    # return b, a
# #end def

# # This function will apply the filter considering the initial transient.
# def butter_highpass_filter_zi(data, highcut, sRate, order=5):
    # b, a = butter_highpass(highcut, sRate, order=order)
    # zi = lfilter_zi(b, a)
    # y,zo = lfilter(b, a, data, zi=zi*data[0])
    # return y
# #end def

# # Butterworth bandpass filter ------------------------------------------------------------
# def butter_bandpass(lowcut, highcut, sRate, order=5):
    # nyq = 0.5 * sRate
    # low = lowcut / nyq
    # high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band')
    # return b, a
# #/def

# # This function will apply the filter considering the initial transient.
# def butter_bandpass_filter_zi(data, lowcut, highcut, sRate, order=5):
    # b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
    # zi = lfilter_zi(b, a)
    # y,zo = lfilter(b, a, data, zi=zi*data[0])
    # return y
# #/def


# Read signals from PPG HUSM -------------------------------------------------------------
def getHUSMppg():

    #print('\nFirst timestamp: ' + str(time_manager.time.getTimestamp()))
    #print('First time: ' + str(time_manager.time.getTime()))

    dataset = 'ppg-dataset_husm'
    list_of_vol = os.listdir(dataset)
    list_of_vol.sort()
		
    records = []
    # print('\nLoading ' + str(dataset) + '\n')
    for type_vol in list_of_vol:
        if (4 <= len(type_vol) <= 6): # To not read trash

            list_of_rec = os.listdir(dataset+'/'+type_vol)
            list_of_rec.sort()
            for volunteer in list_of_rec:
                if (len(volunteer) == 7): # To not read trash

                    list_of_protocol = os.listdir(dataset+'/'+type_vol+'/'+volunteer)
                    list_of_protocol.sort()
                    for protocol in list_of_protocol:
                        if (4 <= len(protocol) <= 8): # To not read trash
                            # print('Getting record ' + protocol + ' from ' + type_vol + ' volunteer number ' + volunteer)
                            # PPG signals file
                            x_ppg, ppg = [], []
                            ppg_base = 1.0
                            sample = 0
                            with open(dataset+'/'+type_vol+'/'+volunteer+"/"+protocol+'/ppg.csv') as data_file:
                                next(data_file)
                                for line in data_file:
                                    aux = line.split(',')
                                    x_ppg.append(int(sample))
                                    ppg.append(ppg_base - float(aux[1])) # ir signal
                                    sample += 1
                            data_file.close()

                            # Beats signals file
                            x_beats, beats = [], []
                            with open(dataset+'/'+type_vol+'/'+volunteer+"/"+protocol+'/beats.csv') as data_file:
                                next(data_file)
                                for line in data_file:
                                    aux = line.split(',')
                                    x_beats.append(float(aux[0]))
                                    beats.append(float(aux[1]))
                            data_file.close()

                            records.append( record(volunteer, x_ppg, ppg, x_beats, beats) )
    return records
import os
import numpy as np
from time_manager import time
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi


# Variables ------------------------------------------------------------------------------
records = []

# Signal Structure -----------------------------------------------------------------------
class record:
    def __init__(self, name, x_ppg, ppg, x_beats, beats):
        self.name = name # Record name
        self.ppg = [x_ppg, ppg] # Record ppg
        self.beats = [x_beats, beats] # Record beats
    #end-def
#end-class


# Butterworth highpass filter ------------------------------------------------------------
def butter_highpass(highcut, sRate, order=5):
    nyq = 0.5 * sRate
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a
#end def

# This function will apply the filter considering the initial transient.
def butter_highpass_filter_zi(data, highcut, sRate, order=5):
    b, a = butter_highpass(highcut, sRate, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y
#end def

# Butterworth bandpass filter ------------------------------------------------------------
def butter_bandpass(lowcut, highcut, sRate, order=5):
    nyq = 0.5 * sRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#/def

# This function will apply the filter considering the initial transient.
def butter_bandpass_filter_zi(data, lowcut, highcut, sRate, order=5):
    b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y
#/def

# Read signals from MIMIC ----------------------------------------------------------------
def getMIMICppg():

    print('\nFirst timestamp: ' + str(time.getTimestamp()))
    print('First time: ' + str(time.getTime()))

    dataset = 'MIMIC1_organized'
    list_of_recs = os.listdir(dataset)

    print('\nLoading ' + str(dataset) + ' dataset\n')
    for item in list_of_recs:
        if (len(item) == 3): # To not read trash

            print('Getting record number ' + item)
    
            # PPG signals file
            x_ppg, ppg = [], []
            with open(dataset+'/'+item+'/record_ppg-ecg.csv') as data_file:
                next(data_file)
                next(data_file)
                for line in data_file:
                    aux = line.split(',')
                    x_ppg.append(int(aux[0]))
                    try:
                        ppg.append(float(aux[1]))
                    except:
                        ppg.append(float(0.0))
                    #/try
                #/for
            #/with

            data_file.close()

            # Beats signals file
            x_beats, beats = [], []
            with open(dataset+'/'+item+'/rri.csv') as data_file:
                next(data_file)
                for line in data_file:
                    aux = line.split(',')
                    shift = int(aux[3])
                    x_beats.append(int(aux[0]) + shift)
                    beats.append(float(aux[1]))
                #/for
            #/with

            data_file.close()

            records.append( record(item, x_ppg, ppg, x_beats, beats) )
        #/if
    #/for
#/def

# Read signals from PPG HUSM -------------------------------------------------------------
def getHUSMppg():

    print('\nFirst timestamp: ' + str(time.getTimestamp()))
    print('First time: ' + str(time.getTime()))

    dataset = 'ppg-dataset_husm/dataset'
    list_of_vol = os.listdir(dataset)

    print('\nLoading ' + str(dataset) + ' dataset\n')
    for type_vol in list_of_vol:
        if (4 <= len(type_vol) <= 6): # To not read trash

            list_of_rec = os.listdir(dataset+'/'+type_vol)
            for volunteer in list_of_rec:
                if (len(volunteer) == 7): # To not read trash

                    list_of_protocol = os.listdir(dataset+'/'+type_vol+'/'+volunteer)
                    for protocol in list_of_protocol:
                        if (4 <= len(protocol) <= 8): # To not read trash

                            print('Getting record ' + protocol + ' from ' + type_vol + ' volunteer number ' + volunteer)


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
                                #/for
                            #/with

                            data_file.close()

                            # Beats signals file
                            x_beats, beats = [], []
                            with open(dataset+'/'+type_vol+'/'+volunteer+"/"+protocol+'/beats.csv') as data_file:
                                next(data_file)
                                for line in data_file:
                                    aux = line.split(',')
                                    x_beats.append(float(aux[0]))
                                    beats.append(float(aux[1]))
                                #/for
                            #/with

                            data_file.close()

                            records.append( record(volunteer, x_ppg, ppg, x_beats, beats) )
                        #/if
                    #/for

                #/if
            #/for

        #/if
    #/for
#/def

# MAIN -----------------------------------------------------------------------------------

# MIMIC dataset
#getMIMICppg()
#print( records[0].name )

# HUSM dataset
getHUSMppg() # get dataset
x = records[0].ppg[0]
ir = records[0].ppg[1]
peakx = records[0].beats[0]
peaky = records[0].beats[1]

# Apply bandpass filter into uPPG raw signals
lowcut = 0.5 # From Elgendi
highcut = 8 # From Elgendi
order = 2 # From Elgendi
sps = 200
ir_f = butter_bandpass_filter_zi(ir, lowcut, highcut, sps, order)

# Plot
plt.figure("PPG and peaks from "+records[0].name,figsize=(14,6))
plt.plot(x, ir_f, color="brown")
plt.scatter(peakx, peaky)
plt.grid()
plt.show()

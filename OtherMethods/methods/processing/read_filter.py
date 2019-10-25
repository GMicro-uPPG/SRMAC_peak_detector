## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
from scipy.signal import butter, lfilter, lfilter_zi


# READ FUNCTIONS -------------------------------------------------------------------------
class read:
    sps = 125
    name = sys.argv[1]
    signal = "MIMIC1_organized/" + name

    # Read signals
    def getSignals():
        x = []
        pleth = []
        ecg = []
        samples = 0

        with open(read.signal + "/record_ppg-ecg.csv") as dataFile:
            next(dataFile)
            next(dataFile)
            for line in dataFile:
                aux = line.split(",")
                x.append(int(aux[0]))
                try:
                    pleth.append(float(aux[1]))
                except:
                    pleth.append(float(0))
                ecg.append(float(aux[2]))
                samples =+ 1
            #/for
        #/with

        dataFile.close()

        return x, pleth, ecg, samples, read.sps, read.name
    #/def

    # Read annotations
    def getAnn():
        x = []
        ann = []
        samples = 0

        with open(read.signal + "/rri.csv") as dataFile:
            next(dataFile)
            for line in dataFile:
                aux = line.split(",")
                shift = int(aux[3])
                x.append(int(aux[0]) + shift)
                ann.append(float(aux[1]))
                samples =+ 1
            #/for
        #/with

        dataFile.close()

        return x, ann, samples, read.name
    #/def
#/class


# FILTER FUNCTIONS -----------------------------------------------------------------------
class bFilter:
    # Bandpass ---
    def butter_bandpass(lowcut, highcut, sRate, order=5):
        nyq = 0.5 * sRate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    #/def

    # This function will apply the filter considering the initial transient.
    def butter_bandpass_filter_zi(data, lowcut, highcut, sRate, order=5):
        b, a = bFilter.butter_bandpass(lowcut, highcut, sRate, order=order)
        zi = lfilter_zi(b, a)
        y,zo = lfilter(b, a, data, zi=zi*data[0])
        return y
    #/def

    # Lowpass ---
    def butter_lowpass(lowcut, sRate, order=5):
        nyq = 0.5 * sRate
        low = lowcut / nyq
        b, a = butter(order, low, btype='low')
        return b, a
    #/def

    # This function will apply the filter considering the initial transient.
    def butter_lowpass_filter_zi(data, lowcut, sRate, order=5):
        b, a = bFilter.butter_lowpass(lowcut, sRate, order=order)
        zi = lfilter_zi(b, a)
        y,zo = lfilter(b, a, data, zi=zi*data[0])
        return y
    #/def
#/class

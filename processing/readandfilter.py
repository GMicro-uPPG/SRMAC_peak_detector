#!/usr/bin/env python3

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal
## PROFESSORS: Cesar Augusto Prior and Cesar Rodrigues (Yeah. Its almost an overflow!)
## PROJECT: Implementation of biomedical signal processing methods
## ARCHIVE: Functions for reading files and applying filters.
## DATE: 06/05/2019 - updated @ 26/05/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


# Read signals ---------------------------------------------------------------------------
def getSignals(uppg=False, psg=False):

    sps = 200
    uPPGsignalBase = 1.045

    signalPath = "raw-signals/" + sys.argv[1] + "/uPPG-PSG-synced/"

    if(uppg):
        plet_redUPPG = []
        plet_irUPPG = []

        # uPPG signals file
        with open(signalPath+"uppg_signals-synced.csv") as dataFile:
            next(dataFile)
            for line in dataFile:
                aux = line.split(",")
                plet_redUPPG.append(uPPGsignalBase - float(aux[0]))
                plet_irUPPG.append(uPPGsignalBase - float(aux[1]))
            #end-for
        #end-with

        dataFile.close()

        if(not psg):
            return plet_redUPPG, plet_irUPPG, sps
    #end-if

    if(psg):
        ecgPSG = []
        pletPSG = []
        annotationPSG = []
        annMarksPSG = []

        # PSG signals file
        index = 0
        with open(signalPath+"psg_signals-synced.csv") as dataFile:
            next(dataFile)
            for line in dataFile:
                aux = line.split(",")
                ecgPSG.append(float(aux[0]))
                pletPSG.append(float(aux[1]))
                annotationPSG.append(float(aux[4]))
                index += 1
                if(float(aux[4]) > 0):
                    annMarksPSG.append(int(index))
                #end-if
            #end-for
        #end-with

        dataFile.close()
        if(not uppg):
            return ecgPSG, pletPSG, annMarksPSG, sps
    #end-if

    if(uppg and psg):
        if(len(plet_redUPPG) != len(ecgPSG)):
            print("Wrong or unsynchronized signal files.")
            return None
        else:
            return plet_redUPPG, plet_irUPPG, ecgPSG, pletPSG, annMarksPSG, sps
        #end-if
    #end-if
#end def


# Butterworth filter ---------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, sRate, order=5):
    nyq = 0.5 * sRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#end-def

# This function will apply the filter considering the initial transient.
def butter_bandpass_filter_zi(data, lowcut, highcut, sRate, order=5):
    b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y
#end-def

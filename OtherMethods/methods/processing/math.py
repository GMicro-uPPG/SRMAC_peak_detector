## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import sys


# MATH FUNCTIONS ----------------------------------------------------------
# Cutting negative values
def cuttingNegatives(signal):
    ssize = len(signal)
    positives = np.zeros(ssize)
    
    for i in np.arange(ssize):
        if(signal[i] > 0):
            positives[i] = signal[i]
    #end-for

    return positives
#end-def

# Squaring values
def squaringValues(signal):
    return np.power(signal,2)
#end-if

# Exponential moving average by abascal. Do not provide delay, but do not calculate first and last values.
def expMovingAverage_abascal(signal, window):
    ssize = len(signal)

    maSum = 0
    expMA = np.zeros(ssize)
    k = int((window-1)/2)

    for i in range(k, ssize-k):
        for j in range(i-k, i+k):
            maSum = maSum + signal[j]
        #end-for
        expMA[i] = maSum/window
        maSum = 0
    #end-for
    
    return expMA
#end-def

# Statistical average
def average(signal):
    ssize = len(signal)

    aSum = 0
    for i in np.arange(ssize):
        aSum = aSum + signal[i]
    
    return aSum/ssize
#end-def

# Calculates blocks of interest with rejected noise, based on the Elgendi method.
def elgendiRealBOIandPeaks(xAxis, signal, MA, THR1, THR2):
    # realBlocksOfInterest = blocks of interest with rejected noise
    samples = len(signal)

    peakx, peaky = [], []
    xpeakmax, ypeakmax, ypeakmaxMAX = 0, 0, 0
    blockWidth = 0
    realBlocksOfInterest = np.zeros(samples)

    for i in np.arange(samples):
        if(MA[i] > THR1[i]):
            blockWidth += 1
            realBlocksOfInterest[i] = 1

            if(signal[i] > ypeakmax):
                xpeakmax = xAxis[i]
                ypeakmax = signal[i]
                if(ypeakmax > ypeakmaxMAX):
                    ypeakmaxMAX = ypeakmax
                #end-if
            #end-if
        elif(blockWidth>=THR2):
            blockWidth = 0
            peakx.append(float(xpeakmax))
            peaky.append(float(ypeakmax))
            xpeakmax = ypeakmax = 0
        #end-if
    #end-for

    # Set blocks area wave amplitude with the maximum ypeak founded.
    realBlocksOfInterest = realBlocksOfInterest * ypeakmaxMAX

    return realBlocksOfInterest, peakx, peaky
#end-def

def billauer_minmax(v, delta, x=None):
    maxtab_x, maxtab_y = [], []
    mintab_x, mintab_y = [], []

    if (x is None):
        x = np.arange(len(v))
    
    v = np.asarray(v)

    if (len(v) != len(x)):
        sys.exit('\nInput vectors v and x must have same lenght.\n')
    
    if (not np.isscalar(delta)):
        sys.exit('\nInput argument delta must be a scalar.\n')

    if (delta <= 0):
        sys.exit('\nInput argument delta must be positive.\n')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]

        if (this > mx):
            mx = this
            mxpos = x[i]
        #/if

        if (this < mn):
            mn = this
            mnpos = x[i]
        #/if

        if (lookformax):
            if (this < (mx-delta)):
                maxtab_x.append(int(mxpos))
                maxtab_y.append(float(mx))
                mn = this
                mnpos = x[i]
                lookformax = False
            #/if
        else:
            if (this > (mn+delta)):
                mintab_x.append(int(mnpos))
                mintab_y.append(float(mn))
                mx = this
                mxpos = x[i]
                lookformax = True
            #/if
        #/if
    #/for

    return maxtab_x, maxtab_y, mintab_x, mintab_y
#/def

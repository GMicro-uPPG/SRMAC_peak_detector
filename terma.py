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

# Author: Cesar Abascal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from read_datasets import records


# TERMA Method ----------------------------------------------------
def peak_positions_terma(ppg_signal):
        """ Detect peak positions using the TERMA method """ 
        def butter_bandpass(lowcut, highcut, sRate, order=5):
            nyq = 0.5 * sRate
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        # def

        # This function will apply the filter considering the initial transient.
        def butter_bandpass_filter_zi(data, lowcut, highcut, sRate, order=5):
            b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
            zi = lfilter_zi(b, a)
            y, zo = lfilter(b, a, data, zi=zi*data[0])
            return y
        # def

        # Cutting negative values
        def cuttingNegatives(signal, ssize):
            positives = np.zeros(ssize)

            for i in np.arange(ssize):
                if(signal[i] > 0):
                    positives[i] = signal[i]
            # for

            return positives
        # def

        # Moving average
        def movingAverage(signal, ssize, window):

            maSum = 0
            mAver = np.zeros(ssize)
            k = int((window-1)/2)

            for i in np.arange(k, ssize-k):
                for j in np.arange(i-k, i+k):
                    maSum = maSum + signal[j]
                # for
                mAver[i] = maSum/window
                maSum = 0
            # for

            return mAver
        # def


        # Statistical average
        def average(signal, ssize):

            aSum = 0
            for i in np.arange(ssize):
                aSum = aSum + signal[i]

            return aSum/ssize
        # end-def


        # Get Systolic Peak Areas
        def getSystolicPeakAreas(nSamples, MApeak, THR1, yf, THR2):
            x = np.arange(0, nSamples)

            # Emphasizing systolic area (block of interest) and finding systolic peaks
            peakx, peaky = [], []
            xpeakmax, ypeakmax, ypeakmaxMAX = 0, 0, 0
            blockWidth = 0
            systolicArea = np.zeros(nSamples)

            for i in np.arange(nSamples):
                if(MApeak[i] > THR1[i]):
                    blockWidth += 1
                    systolicArea[i] = 1

                    if(yf[i] > ypeakmax):
                        xpeakmax = x[i]
                        ypeakmax = yf[i]
                        if(ypeakmax > ypeakmaxMAX):
                            ypeakmaxMAX = ypeakmax
                    # if
                elif(blockWidth >= THR2):
                    blockWidth = 0
                    peakx.append(float(xpeakmax))
                    peaky.append(float(ypeakmax))
                    xpeakmax = ypeakmax = 0
                # if
            # for

            # Set systolic area wave amplitude with the maximum ypeak founded.
            systolicArea = systolicArea * ypeakmaxMAX

            return systolicArea, peakx, peaky
        # def


        # Apply bandpass filter into raw signals
        sRate = 200
        lowcut = 0.5
        highcut = 8
        order = 2
        yf = butter_bandpass_filter_zi(ppg_signal, lowcut, highcut, sRate, order)
        nSamples = len(yf)

        # Clipping signal
        yfc = cuttingNegatives(yf, nSamples)

        # Squaring signal
        yfcs = np.power(yfc, 2)

        # Emphasise the systolic peak area
        # W1 = 111ms (22pts @ 200Hz) correspond to the systolic peak duration
        MApeak = movingAverage(yfcs, nSamples, 22)

        # Emphasise the beat area
        # W2 = 667ms (133pts @ 200Hz) correspond to the heartbeat duration
        MAbeat = movingAverage(yfcs, nSamples, 133)

        # Statiscal mean of the signal
        yfcsa = average(yfcs, nSamples)

        # Alpha will be the multiplication of yfcsa by beta (0.02)
        alpha = 0.02 * yfcsa

        # Threshold1 will be the sum of each point in MAbeat by alpha
        THR1 = MAbeat + alpha  # array

        # Threshold2 will be the same as W1
        THR2 = 22  # scalar

        # Get Systolic Peak Areas
        systolicArea, peakx, peaky = getSystolicPeakAreas(
            nSamples, MApeak, THR1, yf, THR2)

        return peakx
    # def

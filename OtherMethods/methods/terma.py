## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from processing.read_filter import *
from processing.math import *
from processing.plot import elgendi as e


# MAIN -----------------------------------------------------------------------------------
# Read PPG MIMIC1 signals
x, pleth, ecg, samples, sps, name = read.getSignals()

# Apply butter bandpass filter
lowcut = 0.5
highcut = 10
order = 4
pleth_f = bFilter.butter_bandpass_filter_zi(pleth, lowcut, highcut, sps, order)

# Clipping PPG signal
pleth_fc = cuttingNegatives(pleth_f)

# Squaring PPG signal
pleth_fcs = squaringValues(pleth_fc)

# Emphasise the systolic peak area
# W1 = 111ms (13.875pts @ 125Hz) correspond to the systolic peak duration
W1 = 13 # Nearest odd integer
MApeak = expMovingAverage_abascal(pleth_fcs, W1)

# Emphasise the beat area
# W2 = 667ms (83.375pts @ 125Hz) correspond to the heartbeat duration
W2 = 83 # Nearest odd integer
MAbeat = expMovingAverage_abascal(pleth_fcs, W2)

# Statiscal mean of the signal
pleth_fcsa = average(pleth_fcs)

# Alpha will be the multiplication of pleth_fcsa by beta
beta = 0.02 # Provide by Elgendi
alpha = beta * pleth_fcsa

# Threshold1 will be the sum of each point in MAbeat by alpha
THR1 = MAbeat + alpha # array

# Threshold2 will be the same as W1
THR2 = W1 # scalar

# Elgendi BOI
realBlocksOfInterest, peakx, peaky = elgendiRealBOIandPeaks(x, pleth_f, MApeak, THR1, THR2)

# Plot
e.plot(x, pleth_f, peakx, peaky)

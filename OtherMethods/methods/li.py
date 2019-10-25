## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from processing.read_filter import *
from processing.math import *


# MAIN -----------------------------------------------------------------------------------
# Read PPG MIMIC1 signals
x, pleth, ecg, samples, sps, name = read.getSignals()

# Apply butter lowpass filter
lowcut = 0.5
order = 4
pleth_f = bFilter.butter_lowpass_filter_zi(pleth, lowcut, sps, order)

# Apply first derivative on filtered signal
x, pleth_fd = firstDerivative(x, pleth_f)

print(pleth_fd)
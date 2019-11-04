## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from processing.read_filter import *
from processing.math import *

from matplotlib import pyplot as plt

# MAIN -----------------------------------------------------------------------------------
# Read PPG MIMIC1 signals
x, pleth, ecg, samples, sps, name = read.getSignals()

# Apply butter lowpass filter
lowcut = 0.5
order = 3
pleth_f = bFilter.butter_lowpass_filter_zi(pleth, lowcut, sps, order)

# Apply first derivative on filtered signal
x, pleth_fd = firstDerivative(x, pleth_f)


# TERMINAR .......
print(pleth_fd)



plt.figure('Zong Method', figsize=(14,6))
plt.ylabel("Amplitude")
plt.xlabel("Samples")
plt.plot(x, pleth_fd, color='purple')
#plt.scatter(peakx, peaky)
plt.grid()
plt.show()
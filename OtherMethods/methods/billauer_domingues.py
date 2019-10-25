## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from processing.read_filter import *
from processing.math import *
from processing.plot import domingues as d


# MAIN -----------------------------------------------------------------------------------
# Read PPG MIMIC1 signals
x, pleth, ecg, samples, sps, name = read.getSignals()

# Apply Billauer min-max method
delta = 0.5
maxtab_x, maxtab_y, mintab_x, mintab_y = billauer_minmax(pleth, delta, x=None)

# Plot
d.plot(x, pleth, maxtab_x, maxtab_y, mintab_x, mintab_y)

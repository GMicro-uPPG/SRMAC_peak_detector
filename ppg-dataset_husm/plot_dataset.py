import os
import sys
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt


# FUNCTIONS ------------------------------------------------------------------------------
# Read signals ---------------------------------------------------------------------------
def read(signals_path, record_id, peak_id):
    x, ir = [], []
    peakx, peaky = [], []
    ppg_base = 1.0
    sample = 0

    with open(signals_path+record_id) as data_file:
        next(data_file)
        for line in data_file:
            aux = line.split(',')
            x.append(int(sample))
            ir.append(ppg_base - float(aux[1]))
            sample += 1
        #/for
    #/with

    with open(signals_path+peak_id) as data_file:
        next(data_file)
        for line in data_file:
            aux = line.split(',')
            peakx.append(float(aux[0]))
            peaky.append(float(aux[1]))
        #/for
    #/with

    return x, ir, peakx, peaky
#/def

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



# MAIN -----------------------------------------------------------------------------------

# Read
volunteer_type = sys.argv[1]
name = sys.argv[2]
protocol = sys.argv[3]
signals_path = "dataset/" + volunteer_type + "/" + name + "/" + protocol + "/"

x, ir, peakx, peaky = read(signals_path, 'ppg.csv', 'beats.csv')

# Apply bandpass filter into uPPG raw signals
lowcut = 0.5 # From Elgendi
highcut = 8 # From Elgendi
order = 2 # From Elgendi
sps = 200
ir_f = butter_bandpass_filter_zi(ir, lowcut, highcut, sps, order)

# Plot
plt.figure("PPG and peaks",figsize=(14,6))
plt.plot(x, ir_f, color="brown")
plt.scatter(peakx, peaky)
plt.grid()
plt.show()

import os
import numpy as np
from log_manager import time, log


# Signal Structure -----------------------------------------------------------------------
class record:
    def __init__(self, name, x_ppg, ppg, x_hrv, hrv):
        self.name = name # Record name
        self.ppg = [x_ppg, ppg] # Record ppg
        self.hrv = [x_hrv, hrv] # Record hrv
    #end-def
#end-class

# Read signals ---------------------------------------------------------------------------
def getSignals(name):

    signalPath = 'MIMIC1_organized/' + name
    x_ppg, ppg = [], []
    x_hrv, hrv = [], []
   
    # PPG signals file
    with open(signalPath+'/record_ppg-ecg.csv') as dataFile:
        next(dataFile)
        next(dataFile)
        for line in dataFile:
            aux = line.split(',')
            x_ppg.append(int(aux[0]))
            try:
                ppg.append(float(aux[1]))
            except:
                ppg.append(float(0.0))
            #end-try
        #end-for
    #end-with

    dataFile.close()

    # HRV signals file
    with open(signalPath+'/rri.csv') as dataFile:
        next(dataFile)
        for line in dataFile:
            aux = line.split(',')
            shift = int(aux[3])
            x_hrv.append(int(aux[0]) + shift)
            hrv.append(float(aux[1]))
        #end-for
    #end-with

    dataFile.close()

    return name, x_ppg, ppg, x_hrv, hrv
#end-def


# MAIN -----------------------------------------------------------------------------------
records = []
ignore = '.DS_Store'
dataset = 'MIMIC1_organized'
list_of_recs = os.listdir(dataset)

log.openLogFile() # open log file

timestamp = '\nFirst timestamp: ' + str(time.getTimestamp())
log.printWriteLog(timestamp)


dataset_name = '\nLoading ' + str(dataset) + ' dataset\n'
log.printWriteLog(dataset_name)
for i in range(len(list_of_recs)):
    if(list_of_recs[i] != ignore):
        get_rec = 'Getting record number ' + list_of_recs[i]
        log.printWriteLog(get_rec)
        name, x_ppg, ppg, x_hrv, hrv = getSignals(list_of_recs[i])
        records.append( record(name, x_ppg, ppg, x_hrv, hrv) )
    #end-if
#end-for

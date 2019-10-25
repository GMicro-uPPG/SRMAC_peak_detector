#import sys
import matplotlib.pyplot as plt


def getSignals(name):

    #name = sys.argv[1]
    x_ppg, ppg = [], []
    x_hrv, hrv = [], []
   
    # PPG signals file
    with open('MIMIC1_organized_short/' + name + '/record_ppg-ecg.csv') as dataFile:
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
    with open('MIMIC1_organized_short/' + name + '/rri.csv') as dataFile:
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


def plotPPG(name, x_ppg, ppg, x_hrv, hrv):
    plt.figure('PPG Signal ' + str(name) + ' from MIMIC1_organized_short', figsize=(14,6)) # 20,10

    plt.title('PPG Signal')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.plot(x_ppg, ppg, 'purple')
    plt.scatter(x_hrv, hrv)
    plt.grid()

    plt.show()
#end-def


# MAIN ---------------------------------------------------

# name, x_ppg, ppg, x_hrv, hrv = getSignals()
# plotPPG(name, x_ppg, ppg, x_hrv, hrv)
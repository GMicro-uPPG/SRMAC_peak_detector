#!python3

# MIT License

# Copyright (c) 2016 Grupo de Microeletrônica (Universidade Federal de Santa Maria)

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

# Third party
import matplotlib.pyplot as plt


def getSignals(name):

    #name = sys.argv[1]
    x_ppg, ppg = [], []
    x_beats, beats = [], []
   
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
            x_beats.append(int(aux[0]) + shift)
            beats.append(float(aux[1]))
        #end-for
    #end-with

    dataFile.close()

    return name, x_ppg, ppg, x_beats, beats
#end-def


def plotPPG(name, x_ppg, ppg, x_beats, beats):
    plt.figure('PPG Signal ' + str(name) + ' from MIMIC1_organized_short', figsize=(14,6)) # 20,10

    plt.title('PPG Signal')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.plot(x_ppg, ppg, 'purple')
    plt.scatter(x_beats, beats)
    plt.grid()

    plt.show()
#end-def


# MAIN ---------------------------------------------------

# name, x_ppg, ppg, x_beats, beats = getSignals()
# plotPPG(name, x_ppg, ppg, x_beats, beats)
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DEVELOPER: Cesar Abascal and Victor Costa
## PROJECT: 
## ARCHIVE: 
## DATE: 23/09/2019 - updated @ 23/09/2019
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt


class elgendi:
    def plot(x, pleth, peakx, peaky):
        plt.figure('TERMA Method - Signal and peaks', figsize=(14,6))
        plt.ylabel("Amplitude")
        plt.xlabel("Samples")
        plt.plot(x, pleth, color='purple')
        plt.scatter(peakx, peaky)
        plt.grid()
        plt.show()
    #/def
#/class

class domingues:
    def plot(x, pleth, maxtab_x, maxtab_y, mintab_x, mintab_y):
        plt.figure('Domingues Method - Signal and peaks', figsize=(14,6))
        plt.ylabel("Amplitude")
        plt.xlabel("Samples")
        plt.plot(pleth, color='purple')
        plt.scatter(maxtab_x, maxtab_y)
        plt.scatter(mintab_x, mintab_y)
        plt.grid()
        plt.show()
    #/def
#/class
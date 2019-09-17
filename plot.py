import matplotlib.pyplot as plt

def plotPPG(name, ppg, hrv):
    plt.figure('PPG Signal ' + str(name) + ' from MIMIC1', figsize=(14,6)) # 20,10

    plt.title('PPG Signal')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.plot(ppg[0], ppg[1], 'purple')
    plt.scatter(hrv[0], [0.6]*len(hrv[0]))
    plt.grid()

    plt.show()
#end-def
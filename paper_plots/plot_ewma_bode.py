#!python3

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 200.0
alphas = [0.9, 0.8, 0.7]
colors = ['b', 'y', 'g']

signals = []
for alpha in alphas:
    signals.append( signal.dlti([1 - alpha],[1, -alpha], dt = float(1/fs)) )


# Magnitude plots
plt.figure()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

for index, signal in enumerate(signals):
    w, mag, _ = signal.bode()
    # frequency from rad/s to hz
    w /= 2 * np.pi
    plt.semilogx(w, mag, color=colors[index], label = str(alphas[index]), alpha = 0.6)
    
plt.legend()



# Phase plots
plt.figure()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (deg)')

for index, signal in enumerate(signals):
    w, _, phase = signal.bode()
    # frequency from rad/s to hz
    w /= 2 * np.pi
    plt.semilogx(w, phase, color=colors[index], label = str(alphas[index]), alpha = 0.6)
    
plt.legend()



plt.show()
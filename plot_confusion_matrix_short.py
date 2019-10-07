import numpy as np
from ppg_peak_detection import crossover_detector
import plot_short_PPG
import matplotlib.pyplot as plt

signal_name = "039"
_, x_ppg, ppg, x_hrv, _ = plot_short_PPG.getSignals(signal_name)
reference_peaks = np.array(x_hrv) - x_ppg[0]

detector = crossover_detector()
detector.set_parameters(alpha_fast = 0.5144310567075446, alpha_slow = 0.9640168863482605)
_, _, _, detected_peaks = detector.detect_peaks(ppg)

tp, tn, fp, fn, confusion_array = detector.signal_confusion_matrix(detected_peaks, reference_peaks)
print('TP: ', tp, 'TN: ', tn, 'FP: ', fp, 'FN: ', fn)
print('Number of reference peaks: ', len(reference_peaks))
print('Number of reference valleys: ', len(reference_peaks)+1)

plt.figure('PPG Signal ' + signal_name + ' from MIMIC1_organized_short', figsize=(14,6)) # 20,10
plt.title('PPG Signal')
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.plot(ppg, color='k')
plt.scatter(x=reference_peaks, y=[max(ppg)]*len(reference_peaks))

for index, member  in enumerate(confusion_array):
    if member == 'tp':
        index_color = 'g'
    elif member == 'tn':
        index_color = 'b'
    elif member == 'fp':
        index_color = 'r'
    elif member == 'fn':
        index_color = 'gray'
    plt.axvspan(x = index, color = index_color, linewidth = 2, alpha = 0.5)

plt.grid()
plt.show()
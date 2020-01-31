#!python3

import numpy as np
import matplotlib.pyplot as plt

n_samples = 40

def impulse_response(alpha, n):
    if n >= 0:
        return (1 - alpha) * (alpha ** n) 
    else:
        return 0
        
x_axis = np.arange(-10, n_samples)

plt.figure()
plt.xlabel('Samples')

for tup in [[0.9,'b'], [0.8,'y'], [0.7,'g']]:
    alpha = tup[0]
    color = tup[1]
    
    y_axis = []
    for n in x_axis:
        y_axis.append(impulse_response(alpha, n))
        
    # print(x_axis, y_axis)
    plt.step(x_axis, y_axis, where='post')
    plt.plot(x_axis, y_axis, 'C2o', color=color, alpha=0.6, label=str(alpha))

plt.legend()
plt.show()
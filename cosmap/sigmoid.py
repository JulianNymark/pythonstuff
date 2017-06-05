#!/usr/bin/python3.6

import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(t):
    return 1/(1 + math.e**-t)

###########################################

fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')

plt.title('some numbers :^)')

spoints_x, spoints_y = zip(*[(t,sigmoid(t)) for t in np.linspace(-6, 6, 100)])
plt.plot(spoints_x, spoints_y)

plt.show()

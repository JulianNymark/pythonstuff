#!/usr/bin/python3.6

import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

def nor_arr(arr):
    m = max(arr)
    return [x/m for x in arr]

def coseit(x):
    return math.cos(x)

def coseit_arr(arr):
    return [math.cos(x) for x in arr]

def sigmoid(t):
    return 1/(1 + math.e**-t)

###########################################

fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')

plt.title('some numbers')

#plt.savefig('foo.png')
some_waveform = [1, 23, 20, 8, 42, 100, 32, 72, 0, 2, 120, 33, 42, 42]
some_waveform = nor_arr(some_waveform)
# plt.plot(some_waveform)
other_waveform = [x*math.pi for x in range(len(some_waveform))]
other_waveform = [math.cos(x) for x in other_waveform]
# plt.plot(other_waveform)

spoints_x, spoints_y = zip(*[(t,sigmoid(t)) for t in np.linspace(-6, 6, 100)])
plt.plot(spoints_x, spoints_y)

plt.show()

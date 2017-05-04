#!/usr/bin/python3.6

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np

fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')
plt.axis('off')

####################

MAX_BUCKET_SIZE = 4
GRID_SIZE = 101


def topple(m):
    next_m = np.zeros(m.shape)
    for kx, vx in np.ndenumerate(m):
        if vx >= MAX_BUCKET_SIZE:
            m[kx] -= MAX_BUCKET_SIZE
            cool = np.array(kx)
            if kx[1] + 1 <= m.shape[1] - 1:
                next_m[tuple(cool + [0, 1])] += 1
            if kx[1] - 1 >= 0:
                next_m[tuple(cool + [0, -1])] += 1
            if kx[0] + 1 <= m.shape[0] - 1:
                next_m[tuple(cool + [1, 0])] += 1
            if kx[0] - 1 >= 0:
                next_m[tuple(cool + [-1, 0])] += 1
    m += next_m


z = np.ones([int(GRID_SIZE / 2), GRID_SIZE]) * 0
n = np.zeros([int(GRID_SIZE / 2), GRID_SIZE])
n[int(np.round((GRID_SIZE / 2) / 2))][int(np.round(GRID_SIZE / 2))] = 1000

print('z')
print(z)
print('n')
print(n)

z += n

while np.max(z) >= MAX_BUCKET_SIZE:
    # print('before topple!')
    # print(z)
    topple(z)
    # print('after topple!')
    # print(z)

plt.imshow(z, interpolation='nearest')
plt.show()

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import pyopencl as cl
import numpy as np

fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')
plt.axis('off')

###########################

#Read in image

#img = np.random.randn(1080, 1920).astype(np.float32)
img = np.zeros([21, 21]).astype(np.float32)
img[int(21 / 2)][int(21 / 2)] = 4
img[int(21 / 2)][int(21 / 2) + 1] = 4
img[int(21 / 2)][int(21 / 2) + 2] = 4
img[int(21 / 2)][int(21 / 2) + 3] = 4
img[int(21 / 2)][int(21 / 2) + 4] = 4

topplers = np.zeros(img.shape).astype(np.int32)

ctx = cl.create_some_context()

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

# Kernel function (read from file)
with open('kernel.cl', 'r') as myfile:
    src = myfile.read()

# Kernel function instantiation
prg = cl.Program(ctx, src).build()
# Allocate memory for variables on the device
current_g = cl.Buffer(ctx,
                      mf.READ_ONLY | mf.COPY_HOST_PTR,
                      hostbuf=img)
topplers_g = cl.Buffer(ctx,
                       mf.COPY_HOST_PTR,
                       hostbuf=topplers)
next_g = cl.Buffer(ctx,
                   mf.WRITE_ONLY,
                   img.nbytes)
width_g = cl.Buffer(ctx,
                    mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.int32(img.shape[0]))
height_g = cl.Buffer(ctx,
                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                     hostbuf=np.int32(img.shape[1]))


cl.enqueue_fill_buffer(queue, next_g, np.float32(0), 0, topplers.nbytes)

# Call Kernel. Automatically takes care of block/grid distribution
prg.topple(queue, img.shape, None,
           current_g,
           topplers_g,
           next_g,
           width_g,
           height_g)

next_iteration = np.zeros(img.shape).astype(np.float32)
cl.enqueue_copy(queue, next_iteration, next_g)
topplers_r = np.zeros(img.shape).astype(np.int32)
cl.enqueue_copy(queue, topplers_r, topplers_g)

plt.imshow(next_iteration, interpolation='nearest')

plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')
plt.axis('off')

plt.imshow(img, interpolation='nearest')

plt.show()

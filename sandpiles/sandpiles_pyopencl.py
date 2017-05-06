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

topplers = np.zeros([21, 21]).astype(np.int32)

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
img_g = cl.Buffer(ctx,
                  mf.READ_ONLY | mf.COPY_HOST_PTR,
                  hostbuf=img)
topplers_g = cl.Buffer(ctx,
                       mf.COPY_HOST_PTR,
                       hostbuf=topplers)
result_g = cl.Buffer(ctx,
                     mf.WRITE_ONLY,
                     img.nbytes)
width_g = cl.Buffer(ctx,
                    mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.int32(img.shape[0]))
height_g = cl.Buffer(ctx,
                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                     hostbuf=np.int32(img.shape[1]))

# Call Kernel. Automatically takes care of block/grid distribution
prg.topple(queue, img.shape, None,
           img_g,
           topplers_g,
           result_g,
           width_g,
           height_g)
final = np.empty_like(img)
cl.enqueue_copy(queue, final, result_g)

plt.imshow(final, interpolation='nearest')

plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('matplotlib_float_')
plt.axis('off')

plt.imshow(img, interpolation='nearest')

plt.show()

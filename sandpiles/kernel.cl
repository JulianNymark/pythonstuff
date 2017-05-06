void topple(__global float *current, __global int *topplers, __global float *next, __global int *width, __global int *height) {
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;

    next[i] = 0; // clear all
    topplers[i] = 0; // clear all

    if (current[i] >= 4.0f) {
        topplers[i] = 1; // set some
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int ix0, ix1, iy0, iy1;
    ix0 = i - 1;
    ix1 = i + 1;
    iy0 = i - w;
    iy1 = i + w;

    // if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 ) {

    //// add toppler neighbors
    if (topplers[ix0] >= 1 && (posx - 1) >= 0) {
        next[i] += 1.0f;
    }
    if (topplers[ix1] >= 1 && (posx + 1) < w) {
        next[i] += 1.0f;
    }
    if (topplers[iy0] >= 1 && (posy - 1) >= 0) {
        next[i] += 1.0f;
    }
    if (topplers[iy1] >= 1 && (posy + 1) < h) {
        next[i] += 1.0f;
    }
    //next[i] += topplers[ix0] + topplers[ix1] + topplers[iy0] + topplers[iy1];

    if (topplers[i] >= 1) {
        //// subtract topplers self
        next[i] -= 4.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    current[i] += next[i];
}

__kernel void toppleKernel(__global float *current, __global int *topplers,  __global float *next, __global int *width, __global int *height) {
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;

    // TODO: sick while loop here (just have all write if not 0, and check if not 0)
    // while any(current) > 4:

    while (current[i] >= 4.0f) {
        /* barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); */
        /* if (posx == 0 && posy == 0) { */
        topple(current, topplers, next, width, height);
        /* } */
    }
}

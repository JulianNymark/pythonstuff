__kernel void topple(__global float *img, __global int *topplers,  __global float *result, __global int *width, __global int *height) {
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;

    //// mark topplers
    if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 ) {
        // at edge = you never topple (TODO, do the edge cases :poop:)
        topplers[i] = 0;
    } else {
        if (img[i] >= 4) {
            topplers[i] = 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int ix0, ix1, iy0, iy1;
    ix0 = i - 1;
    ix1 = i + 1;
    iy0 = i - w;
    iy1 = i + w;

    //// add toppler neighbors
    if (topplers[ix0] >= 1) {
        result[i] += 1;
    }
    if (topplers[ix1] >= 1) {
        result[i] += 1;
    }
    if (topplers[iy0] >= 1) {
        result[i] += 1;
    }
    if (topplers[iy1] >= 1) {
        result[i] += 1;
    }

    if (topplers[i] >= 1) {
        //// subtract topplers self
        result[i] -= 4;
    }
}

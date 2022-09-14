#include <cuda.h>
#include "image.h"

#ifdef _WIN32
    #include <time.h>
    #include <sys/timeb.h>
#else
    #include <sys/time.h>
    #include <sys/timeb.h>
#endif

#define DIM 512
#define PI 3.1415926535f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.001f
#define SPEED 0.25f

__global__ void copy_const_kernel(float *iptr, const float *cptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) {
        iptr[offset] = cptr[offset];
    }
}

__global__ void blend_kernel(float *outSrc, const float* inSrc) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0) {
        left++;
    }
    if (x == DIM - 1) {
        right--;
    }

    int up = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0) {
        up += DIM;
    }
    if (y == DIM - 1) {
        bottom -= DIM;
    }

    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[up] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}

__global__ void float_to_color(unsigned char *bitmap, float* thermal_grid) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    bitmap[offset * 4 + 0] = (int)(255 * thermal_grid[offset]);
    bitmap[offset * 4 + 1] = (int)(255 * thermal_grid[offset]);
    bitmap[offset * 4 + 2] = (int)(255 * thermal_grid[offset]);
    bitmap[offset * 4 + 3] = 255;
}

void anim_gpu(DataBlock *d, int ticks = 90) {
    cudaEventRecord(d->start, 0);
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16); 
    IMAGE<float> *bitmap = d->bitmap;

    for (int i = 0; i < ticks; i++) {
        copy_const_kernel<<<grids, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<grids, threads>>>(d->dev_outSrc, d->dev_inSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }

    float_to_color<<<grids, threads>>>(d->output_bitmap, d->dev_inSrc);

    cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

    cudaEventRecord(d->stop, 0);
    cudaEventSynchronize(d->stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
    
    d->totalTime = elapsedTime;
    ++d->frams;
    printf("Time in frame %d to generate: %3.6f ms\n", d->frams, d->totalTime);
    printf("Average time per tick(%d ticks) to generate: %3.6f ms\n", ticks, d->totalTime / ticks);
}

void anim_exit(DataBlock *d) {
    cudaFree(d->dev_constSrc);
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);

    cudaEventDestroy(d->start);
    cudaEventDestroy(d->stop);
}

int main(int argc, char *argv[]) {
    struct timeb timeSeed;
    ftime(&timeSeed);
    srand(timeSeed.time * 1000 + timeSeed.millitm);

    DataBlock data;
    IMAGE<float> bitmap(DIM, DIM);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frams = 0;
    cudaEventCreate(&data.start);
    cudaEventCreate(&data.stop);

    cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
    cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

    float *temp_const = (float*)malloc(bitmap.image_size());
    float *temp_in = (float*)malloc(bitmap.image_size());
    for (int i = 0; i < DIM * DIM; i++) {
        float flag = (float)(rand() % 101) / 100.0f;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 200) && (x < 400) && (y > 310) && (y < 400)) {
            temp_const[i] = (flag > 0.96f)?MAX_TEMP:0;
        } else {
            temp_in[i] = (flag < 0.1f)?flag:0;
        }
    }

    cudaMemcpy(data.dev_constSrc, temp_const, bitmap.image_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(data.dev_inSrc, temp_in, bitmap.image_size(), cudaMemcpyHostToDevice);

    free(temp_const);
    free(temp_in);

    // anim_gpu(&data);
    // data.bitmap->show_image();
    typedef void (*opt)(DataBlock*, int);
    data.bitmap->show_animations<opt, DataBlock>(anim_gpu, &data, 50);
    anim_exit(&data);

    return 0;
}
#include <cuda_runtime.h>
#include "../inc/image.cuh"
#include "../inc/block.cuh"

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

texture<float, 2> textIn;
texture<float, 2> textOut;
texture<float, 2> textConst;

__global__ void copy_const_kernel(float *iptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(textConst, x, y);
    if (c != 0) {
        iptr[offset] = c;
    }
}

/**
 * @brief 
 * 
 * @param outSrc data destination
 * @param dstOut is the out source the devOut 
 */
__global__ void blend_kernel(float *outSrc, bool dstOut) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;
    // t = (dstOut)?tex2D(textIn, x, y - 1):tex2D(textOut, x, y - 1);
    // l = (dstOut)?tex2D(textIn, x - 1, y):tex2D(textOut, x - 1, y);
    // c = (dstOut)?tex2D(textIn, x, y):tex2D(textOut, x, y);
    // r = (dstOut)?tex2D(textIn, x + 1, y):tex2D(textOut, x + 1, y);
    // b = (dstOut)?tex2D(textIn, x, y + 1):tex2D(textOut, x, y + 1);
    if (dstOut) {
        t = tex2D(textIn, x, y - 1);
        l = tex2D(textIn, x - 1, y);
        c = tex2D(textIn, x, y);
        r = tex2D(textIn, x + 1, y);
        b = tex2D(textIn, x, y + 1);
    } else {
        t = tex2D(textOut, x, y - 1);
        l = tex2D(textOut, x - 1, y);
        c = tex2D(textOut, x, y);
        r = tex2D(textOut, x + 1, y);
        b = tex2D(textOut, x, y + 1);
    }

    outSrc[offset] = c + SPEED * (t + b + l + r - c * 4);
}

__global__ void float_to_color(unsigned char *bitmap, bool dstOut) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = (dstOut)?tex2D(textOut, x, y):tex2D(textIn, x, y);
    bitmap[offset * 4 + 0] = (int)(255 * c);
    bitmap[offset * 4 + 1] = (int)(255 * c);
    bitmap[offset * 4 + 2] = (int)(255 * c);
    bitmap[offset * 4 + 3] = 255;
}

void anim_gpu(DataBlock *d, int ticks = 90) {
    cudaEventRecord(d->start, 0);
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16); 
    IMAGE<float> *bitmap = d->bitmap;

    volatile bool dstOut = true;
    for (int i = 0; i < ticks; i++) {
        float *in, *out;
        in = (dstOut)?d->dev_inSrc:d->dev_outSrc;
        out = (dstOut)?d->dev_outSrc:d->dev_inSrc;
        copy_const_kernel<<<grids, threads>>>(in);
        blend_kernel<<<grids, threads>>>(out, dstOut);
        dstOut = !dstOut;
    }

    float_to_color<<<grids, threads>>>(d->output_bitmap, !dstOut);

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
    cudaUnbindTexture(textIn);
    cudaUnbindTexture(textOut);
    cudaUnbindTexture(textConst);
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
    IMAGE<float> bitmap = IMAGE<float>(DIM, DIM, "head conductions");
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frams = 0;
    cudaEventCreate(&data.start);
    cudaEventCreate(&data.stop);

    cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
    cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
    cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, textIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM);
    cudaBindTexture2D(NULL, textOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM);
    cudaBindTexture2D(NULL, textConst, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM);

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
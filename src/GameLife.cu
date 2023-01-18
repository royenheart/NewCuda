#include "../inc/image.cuh"
#include "../inc/block.cuh"
#include <cstdlib>
#include <cuda_runtime.h>

#define DIM 512

__global__ void rule(bool *in, bool *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int countAlive = 0;
    int lu, ll, lb, u, b, ru, rr, rb;
    int *ele[8];
    lu = offset - 1 - DIM; ele[0] = &lu;
    ll = offset - 1; ele[1] = &ll;
    lb = offset - 1 + DIM; ele[2] = &lb;
    u = offset - DIM; ele[3] = &u;
    b = offset + DIM; ele[4] = &b;
    ru = offset + 1 - DIM; ele[5] = &ru;
    rr = offset + 1; ele[6] = &rr;
    rb = offset + 1 + DIM; ele[7] = &rb;

    if (x == 0) {
        ll += 1;
        lb += 1;
        lu += 1;
    }
    if (x == DIM) {
        rr -= 1;
        rb -= 1;
        ru -= 1;
    }
    if (y == 0) {
        lu += DIM;
        u += DIM;
        ru += DIM;
    }
    if (y == DIM) {
        lb -= DIM;
        b -= DIM;
        rb -= DIM;
    }

    for (int i = 0; i < 8; i++) {
        int num = *ele[i];
        if (num != offset && in[num]) {
            countAlive++;
        }
    }

    if (countAlive > 3 || countAlive < 2) {
        out[offset] = false;
    } else if (countAlive == 3) {
        out[offset] = true;
    }
}

__global__ void float_to_color(unsigned char *bitmap, bool *map) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int c = map[offset]?255:0;
    bitmap[offset * 4 + 0] = c;
    bitmap[offset * 4 + 1] = c;
    bitmap[offset * 4 + 2] = c;
    bitmap[offset * 4 + 3] = 255;
}

void anim_gpu(BoolBlock *d, int ticks = 90) {
    cudaEventRecord(d->start, 0);
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    IMAGE<bool> *bitmap = d->bitmap;

    for (int i = 0; i < ticks; i++) {
        rule<<<grids, threads>>>(d->dev_inSrc, d->dev_outSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }

    float_to_color<<<grids, threads>>>(d->output_bitmap, d->dev_inSrc);
    cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->resultImage_size(), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(d->stop, 0);
    cudaEventSynchronize(d->stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
    
    d->totalTime = elapsedTime;
    ++d->frams;
    printf("Time in frame %d to generate: %3.6f ms\n", d->frams, d->totalTime);
    printf("Average time per tick(%d ticks) to generate: %3.6f ms\n", ticks, d->totalTime / ticks);
}

void anim_free(BoolBlock *d) {
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->output_bitmap);
    free(d->bitmap);
}

int main() {
    BoolBlock data;
    IMAGE<bool> image = IMAGE<bool>(DIM, DIM, "Game of Life");
    data.bitmap = &image;
    data.totalTime = 0;
    data.frams = 0;

    cudaEventCreate(&data.start);
    cudaEventCreate(&data.stop);

    cudaMalloc((void**)&data.output_bitmap, image.resultImage_size());
    cudaMalloc((void**)&data.dev_inSrc, image.image_size());
    cudaMalloc((void**)&data.dev_outSrc, image.image_size());

    cudaMemset(data.dev_inSrc, false, image.image_size());

    bool *temp_s = (bool*)malloc(image.image_size());
    memset(temp_s, false, image.image_size());
    for (int i = 0; i < DIM * DIM; i++) {
        float flag = (float)(rand() % 101) / 100.0f;
        int x = i % DIM;
        int y = i / DIM;
        if (flag > 0.5f) {
            temp_s[i] = true;
        }
    }
    cudaMemcpy(data.dev_inSrc, temp_s, image.image_size(), cudaMemcpyHostToDevice);
    free(temp_s);

    typedef void (*opt)(BoolBlock*, int);
    data.bitmap->show_animations<opt, BoolBlock>(anim_gpu, &data, 1);
    anim_free(&data);

    return 0;
}
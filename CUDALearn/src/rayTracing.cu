/**
 * @file rayTracing.cu
 * @author your name (you@domain.com)
 * @brief 测试三种情况：1. 在GPU直接开辟内存使用（非静态，全局内存） 2. 常量内存
 * @version 0.1
 * @date 2022-08-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cuda.h>
#include "image.h"
#include <cstdlib>

#ifdef _WIN32
    #include <time.h>
    #include <sys/timeb.h>
#else
    #include <sys/time.h>
    #include <sys/timeb.h>
#endif

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 200
#define DIM 1024

struct sphere{
    float r,g,b;
    float radius;
    float x,y,z;
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

// __constant__ sphere s[SPHERES];

__global__ void kernel(sphere *s, unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float maxz = -INF;
    float r = 0, g = 0, b = 0;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float z = s[i].hit(ox, oy, &n);
        if (z > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = z;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

int main(int argc, char **argv) {
    sphere *s;

    struct timeb timeSeed;
    ftime(&timeSeed);
    srand(timeSeed.time * 1000 + timeSeed.millitm);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    IMAGE<float> image(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc((void**)&dev_bitmap, image.image_size());
    cudaMalloc((void**)&s, sizeof(sphere) * SPHERES);
    
    sphere* temp_s = (sphere*)malloc(sizeof(sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    cudaMemcpyToSymbol(s, temp_s, sizeof(sphere) * SPHERES);
    cudaMemcpy(s, temp_s, sizeof(sphere) * SPHERES, cudaMemcpyHostToDevice);
    free(temp_s);

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(s, dev_bitmap);

    cudaMemcpy(image.get_ptr(), dev_bitmap, image.image_size(), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to generate: %3.6f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    image.show_image();
    
    cudaFree(s);
    cudaFree(dev_bitmap);
    
}
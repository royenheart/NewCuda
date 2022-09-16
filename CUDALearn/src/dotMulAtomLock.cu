/**
 * @file dotMulAtomLock.cu
 * @author RoyenHeart (royenheart@outlook.com)
 * @brief 基于原子锁的点积操作
 * @version 0.1
 * @date 2022-08-19
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cstdlib>
#include <iostream>
#include <cuda.h>

#define imin(a,b) (a<b?a:b)
#define N 20 * 1024 * 1024
#define THREADS_PER_BLOCK 256
#define sum_square(x) (x*(x+1)*(2*x+1)/6)

const int blocksPerGrid = imin(32,(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

typedef struct Lock Lock;

struct Lock {
    int *mutex;
    Lock(void) {
        int state = 0;
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }

    ~Lock(void) {
        cudaFree(mutex);
    }

    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    __device__ void unlock(void) {
        atomicExch(mutex, 0);
    }
};

__global__ void dotMul(Lock lock, float *a, float *b, float *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    __shared__ float caches[THREADS_PER_BLOCK];
    
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    caches[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            caches[cacheIndex] += caches[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        // 互斥锁，调用device核函数
        lock.lock();
        *c += caches[0];
        lock.unlock();
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *a, *b, c = 0;
    float *deva, *devb, *devc;
    float elaspedT;
    Lock lock;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);  

    cudaMalloc((void**)&deva, sizeof(float) * N);
    cudaMalloc((void**)&devb, sizeof(float) * N);
    cudaMalloc((void**)&devc, sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaEventRecord(start, 0);

    cudaMemcpy(deva, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(devc, &c, sizeof(float), cudaMemcpyHostToDevice);

    dotMul<<<blocksPerGrid, THREADS_PER_BLOCK>>>(lock, deva, devb, devc);

    cudaMemcpy(&c, devc, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elaspedT, start, stop);
    
    printf("Time: %3.6f ms\n", elaspedT);
    printf("Value %3.6f = %3.6f ?\n", c, 2 * sum_square((float)(N - 1)));

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    free(a);
    free(b);

    return 0;
}
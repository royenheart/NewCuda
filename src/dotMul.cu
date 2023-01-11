#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define imin(a,b) (a<b?a:b)
#define MAX_ELEMENTS 32 * 1024
#define MAX_FLOAT_ELE MAX_ELEMENTS * sizeof(float)
#define THREADS_PER_BLOCK 256
#define sum_square(x) (x*(x+1)*(2*x+1)/6)

const int blocksPerGrid = imin(32,(MAX_ELEMENTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

using namespace std;

__global__ void dotMul(float *a, float *b, float *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    __shared__ float caches[THREADS_PER_BLOCK];
    
    float temp = 0;
    while (tid < MAX_ELEMENTS) {
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
        c[blockIdx.x] = caches[0];
    }
}

int main(int argc, char *argv[]) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc(MAX_FLOAT_ELE);
    b = (float*)malloc(MAX_FLOAT_ELE);
    partial_c = (float*)malloc(MAX_FLOAT_ELE);

    cudaMalloc((void**) &dev_a, MAX_FLOAT_ELE);
    cudaMalloc((void**) &dev_b, MAX_FLOAT_ELE);
    cudaMalloc((void**) &dev_partial_c, blocksPerGrid * sizeof(float));

    for (int i = 0; i < MAX_ELEMENTS; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy(dev_a, a, MAX_FLOAT_ELE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, MAX_FLOAT_ELE, cudaMemcpyHostToDevice);

    dotMul<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    printf("Does CPU value %.6g = %.6g ?\n", c, 2 * sum_square( (float)(MAX_ELEMENTS - 1) ));
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
}
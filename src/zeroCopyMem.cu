#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define N 32 * 1024 * 1024
#define imin(a,b) (a<b?a:b)
#define THREADS_PER_BLOCK 256

const int BLOCKS_PER_GRID = imin(32, (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);

__global__ void dotMul(float *a, float *b, float *c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    __shared__ float caches[THREADS_PER_BLOCK];
    
    float temp = 0;
    while (tid < size) {
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

float malloc_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, *c;
    float result;
    float *deva, *devb, *devc;
    float elapsedT;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    c = (float*)malloc(size * sizeof(float));

    cudaMalloc((void**)&deva, size * sizeof(float));
    cudaMalloc((void**)&devb, size * sizeof(float));
    cudaMalloc((void**)&devc, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i + 2;
    }

    cudaEventRecord(start, 0);

    cudaMemcpy(deva, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, size * sizeof(float), cudaMemcpyHostToDevice);

    dotMul<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(deva, devb, devc, size);

    cudaMemcpy(c, devc, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedT, start, stop);
    
    result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += c[i];
    }

    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    free(a);
    free(b);
    free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Value calculated: %3.6f\n", result);
    return elapsedT;
}

float cuda_host_alloc_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, *c;
    float result;
    float *deva, *devb, *devc;
    float elapsedT;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&c, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i + 2;
    }

    cudaHostGetDevicePointer(&deva, a, 0);
    cudaHostGetDevicePointer(&devb, b, 0);
    cudaHostGetDevicePointer(&devc, c, 0);

    cudaEventRecord(start, 0);

    dotMul<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(deva, devb, devc, size);

    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedT, start, stop);
    
    result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += c[i];
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Value calculated: %3.6f\n", result);
    return elapsedT;
}

float cuda_host_alloc_H2D_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, *c;
    float result;
    float *deva, *devb, *devc;
    float elapsedT;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    c = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i + 2;
    }

    cudaHostGetDevicePointer(&deva, a, 0);
    cudaHostGetDevicePointer(&devb, b, 0);
    cudaMalloc((void**)&devc, size * sizeof(float));

    cudaEventRecord(start, 0);

    dotMul<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(deva, devb, devc, size);

    cudaThreadSynchronize();

    cudaMemcpy(c, devc, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedT, start, stop);
    
    result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += c[i];
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    free(c);
    cudaFree(devc);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Value calculated: %3.6f\n", result);
    return elapsedT;
}

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (prop.canMapHostMemory != 1) {
        printf("Device can't map memory!\n");
        return 0;
    }
    printf("Cuda Integreted: %s\n", (prop.integrated == 1)?"True":"False");

    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    float elapsedTNoMapped = malloc_test(N);
    printf("Elapsed No Mapped: %3.6f ms\n", elapsedTNoMapped);
    
    float elapsedTMapped = cuda_host_alloc_test(N);
    printf("Elapsed Mapped: %3.6f ms\n", elapsedTMapped);

    float elapsedTMappedH2D = cuda_host_alloc_H2D_test(N);
    printf("Elapsed Mapped H2D: %3.6f ms\n", elapsedTMappedH2D);

    return 0;
}

#include "image.h"
#include <cstdlib>
#include "time.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo) {
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();
    
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd(&(temp[buffer[i]]), 1);
        i += stride;
    }

    __syncthreads();
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main() {
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    unsigned int histo_host[256];
    unsigned char *dev_buffer;
    unsigned int *dev_histo;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMallocManaged((void**)&dev_buffer, SIZE);
    cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);
    cudaMallocManaged((void**)&dev_histo, 256 * sizeof(unsigned int));
    cudaMemset(dev_histo, 0, 256 * sizeof(unsigned int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
    int threads = prop.maxThreadsPerBlock;

    // Now start the kernel
    histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time: %3.6f ms\n", elapsedTime);

    // Copy the kernel final out to the host
    // cudaMemcpy(histo_host, dev_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost); 

    // Now verify the data

    long countSize = 0;
    for (int i = 0; i < 256; i++) {
        countSize += dev_histo[i];
    }
    printf("device cal count size %d\n", countSize);

    for (unsigned int i = 0; i < SIZE; i++) {
        dev_histo[buffer[i]]--;
    }
    for (int i = 0; i < 256; i++) {
        if (dev_histo[i] != 0) {
            printf("Result failed! in location %d\n", i);
            break;
        }
    }

    free(buffer);

    cudaFree(dev_buffer);
    cudaFree(dev_histo);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
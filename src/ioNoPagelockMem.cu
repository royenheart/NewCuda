#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define SIZE (1024 * 1024) 

void malloc_test(bool up = true) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *host, *dev;

    host = (float*)malloc(sizeof(float) * SIZE);
    cudaMalloc((void**)&dev, sizeof(float) * SIZE);

    int N = 100;
    if (up) {
        for (int i = 0; i < N; i++) {
            cudaMemcpy(dev, host, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
        }
    } else {
        for (int i = 0; i < N; i++) {
            cudaMemcpy(host, dev, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedT;
    cudaEventElapsedTime(&elapsedT, start, stop);

    printf("Elapsed Time total: %.6f ms\n", elapsedT);
    printf("%s Transport speed: %.6fMB/s\n", (up)?"H2D":"D2H",(float)(100 * sizeof(float) * SIZE) / 1024.0f / 1024.0f * 1000.0f / elapsedT);

    free(host);
    cudaFree(dev);
    cudaFree(start);
    cudaFree(stop);
}

int main() {
    malloc_test(true);
    malloc_test(false);

    return 0;
}
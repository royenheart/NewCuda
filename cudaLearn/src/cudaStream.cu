#include <cstdlib>
#include <iostream>
#include <cuda.h>

#ifdef _WIN32
    #include <time.h>
    #include <sys/timeb.h>
#else
    #include <sys/time.h>
    #include <sys/timeb.h>
#endif

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)
#define INT_N (N * sizeof(int))
#define INT_FULL_DATA_SIZE (FULL_DATA_SIZE * sizeof(int))

using namespace std;

__global__ void kernel(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2.0f;
    }
}

int main(int argc, char* argv[]) {
    struct timeb timeSeed;
    ftime(&timeSeed);
    srand(timeSeed.time * 1000 + timeSeed.millitm);

    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        cout << "Device can't use device overlap, exit!" << endl;
        return 0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *hosta, *hostb, *hostc;
    int *deva, *devb, *devc;

    cudaMalloc((void**)&deva, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devb, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devc, INT_FULL_DATA_SIZE);

    cudaHostAlloc((void**)&hosta, INT_FULL_DATA_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostb, INT_FULL_DATA_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostc, INT_FULL_DATA_SIZE, cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        hosta[i] = rand();
        hostb[i] = rand();
    }

    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        cudaMemcpyAsync(deva, hosta + i, INT_N, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devb, hostb + i, INT_N, cudaMemcpyHostToDevice, stream);
        kernel<<<N / 256, 256, 0, stream>>>(deva, devb, devc);
        cudaMemcpyAsync(hostc + i, devc, INT_N, cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedT; 
    cudaEventElapsedTime(&elapsedT, start, stop);
    printf("Time taken: %3.6f ms\n", elapsedT);

    cudaFreeHost(hosta);
    cudaFreeHost(hostb);
    cudaFreeHost(hostc);
    cudaFree(deva);
    cudaFree(devb);
    cudaFree(devc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}
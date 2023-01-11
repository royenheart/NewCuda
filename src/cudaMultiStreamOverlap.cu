#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

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

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int *hosta, *hostb, *hostc;
    int *deva1, *deva2, *devb1, *devb2, *devc1, *devc2;

    cudaMalloc((void**)&deva1, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devb1, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&deva2, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devb2, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devc1, INT_FULL_DATA_SIZE);
    cudaMalloc((void**)&devc2, INT_FULL_DATA_SIZE);

    cudaHostAlloc((void**)&hosta, INT_FULL_DATA_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostb, INT_FULL_DATA_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostc, INT_FULL_DATA_SIZE, cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        hosta[i] = rand();
        hostb[i] = rand();
    }

    for (int i = 0; i < FULL_DATA_SIZE; i += 2 * N) {
        cudaMemcpyAsync(deva1, hosta + i, INT_N, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(deva2, hosta + i + N, INT_N, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(devb1, hostb + i, INT_N, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(devb2, hostb + i + N, INT_N, cudaMemcpyHostToDevice, stream2);
        kernel<<<N / 256, 256, 0, stream1>>>(deva1, devb1, devc1);
        kernel<<<N / 256, 256, 0, stream2>>>(deva2, devb2, devc2);
        cudaMemcpyAsync(hostc + i, devc1, INT_N, cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(hostc + i + N, devc2, INT_N, cudaMemcpyDeviceToHost, stream2);
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedT; 
    cudaEventElapsedTime(&elapsedT, start, stop);
    printf("Time taken: %3.6f ms\n", elapsedT);

    cudaFreeHost(hosta);
    cudaFreeHost(hostb);
    cudaFreeHost(hostc);
    cudaFree(deva1);
    cudaFree(devb1);
    cudaFree(devc1);
    cudaFree(deva2);
    cudaFree(devb2);
    cudaFree(devc2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
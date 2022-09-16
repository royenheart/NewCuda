#pragma once

#include <string>

#ifdef _WIN32
    #include <time.h>
#else
    #include <sys/time.h>
#endif

#define MAX_CARDS 64

using namespace std;

/**
 * @brief Get every available Cards Infomation
 * 
 */
void getCardInfo() {
    cudaDeviceProp prop;

    int Count;
    cudaGetDeviceCount(&Count);
    for (int i = 0; i < Count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("--- Card %d Info: ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels);
        printf("--- Card %d Info: ---\n", i);
    }
}
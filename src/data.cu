#include "../inc/data.cuh"
#include <cstdlib>

void* big_random_block(int size) {
    unsigned char *data = (unsigned char*)malloc( size );
    for (int i=0; i<size; i++) {
        data[i] = rand();
    }

    return data;
}
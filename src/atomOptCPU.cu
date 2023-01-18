#include <cstdlib>
#include <cstdio>
#include "../inc/data.cuh"

#ifdef _WIN32
    #include <time.h>
    #include <sys/timeb.h>
#else
    #include <sys/time.h>
    #include <sys/timeb.h>
#endif

#define SIZE (100 * 1024 * 1024)

int main(int argc, char* argv[]) {
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    unsigned int histo[256];
    memset(histo, 0, 256 * sizeof(unsigned int));
    clock_t start, stop;
    start = clock();

    for (unsigned int i = 0; i < SIZE; i++) {
        histo[buffer[i]]++;
    }

    long coutSize = 0;
    for (int i = 0; i < 256; i++) {
        coutSize += histo[i];
    }

    stop = clock();
    printf("Time: %.6lf ms\n", (double)(stop - start) / CLOCKS_PER_SEC * 1000);
    printf("count size %ld\n", coutSize);
    free(buffer);
    return 0;
}
#include "book.h"
#include <cstdlib>
#include "time.h"

#define SIZE (100 * 1024 * 1024)

int main() {
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
    printf("count size %d\n", coutSize);
    free(buffer);
    return 0;
}
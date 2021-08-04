#ifndef __ADELTIMER_H
#define __ADELTIMER_H

#include <time.h>
#include <inttypes.h>
// #include <stdint-uintn.h>

clock_t startTime;
struct timespec start, end;


inline  __attribute__((always_inline)) void startTimer(void) {
    // startTime = clock();
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    
}


inline __attribute__((always_inline)) int endTimer(void) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    //uint64_t delta_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    //return delta_ms;
    uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    return delta_us;
    // clock_t diff = clock() - startTime;
    // int msec = diff * 1000 / CLOCKS_PER_SEC;
}


#endif //__ADELTIMER_H

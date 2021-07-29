#ifndef __ADELTIMER_H
#define __ADELTIMER_H

#include <time.h>

clock_t startTime;


inline  __attribute__((always_inline)) void startTimer(void) {
    startTime = clock();
}


inline __attribute__((always_inline)) int endTimer(void) {
    clock_t diff = clock() - startTime;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    return msec;
}


#endif //__ADELTIMER_H
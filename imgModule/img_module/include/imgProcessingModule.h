#ifndef IMGPROCESSINGMODULE_H
#define IMGPROCESSINGMODULE_H


void
threshold (unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void 
thresholdUnrolled(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);


void 
thresholdFast(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void 
thresholdSSE2(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void thresholdParallel(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

#endif
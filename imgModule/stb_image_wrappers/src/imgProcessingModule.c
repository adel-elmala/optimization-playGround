#include <stdio.h>
#include <emmintrin.h>

typedef struct{
    unsigned char grey;
    unsigned char alpha;
} pixel2;


typedef struct{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel3;

typedef struct{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel4;




void threshold(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue)
{
    // loop rows
    for(int i = 0; i < width; ++i){
        
        int rowStride = i * height * channels;

        // loop cols
        for (int j = 0 ; j < height * channels ; ++j){
            
            unsigned char pVal = srcData[rowStride + j];
            dstData[rowStride + j] = (pVal < thresholdValue)? 0 : pVal ;
        }
    }
}


void thresholdUnrolled(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue)
{
    // loop rows
    for(int i = 0; i < width; ++i){
        
        int rowStride = i * height * channels;

        // loop cols
        for (int j = 0 ; j < height * channels ; j+=4 ){
            
            unsigned char pVal1 = srcData[rowStride + j];
            unsigned char pVal2 = srcData[rowStride + j + 1];
            unsigned char pVal3 = srcData[rowStride + j + 2];
            unsigned char pVal4 = srcData[rowStride + j + 3];

            dstData[rowStride + j]     = (pVal1 < thresholdValue)? 0 : pVal1 ;
            dstData[rowStride + j + 1] = (pVal2 < thresholdValue)? 0 : pVal2 ;
            dstData[rowStride + j + 2] = (pVal3 < thresholdValue)? 0 : pVal3 ;
            dstData[rowStride + j + 3] = (pVal4 < thresholdValue)? 0 : pVal4 ;
        }
    }
}


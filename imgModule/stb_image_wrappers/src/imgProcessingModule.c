#include <stdio.h>


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




void threshold(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue){

    
    // loop rows
    for(int i = 0; i < width; ++i){
        
        int rowStride = i * height * channels;

        // loop cols
        for (int j = 0 ; j < height * channels ; ++j){
            
            unsigned char pVal = srcData[rowStride + j];
            // if(pVal < thresholdValue)
            //     dstData[rowStride + j] = 0;
            // else
            //     dstData[rowStride + j] = pVal;
            dstData[rowStride + j] = (pVal < thresholdValue)? 0 : pVal ;
            


        }
    }





}




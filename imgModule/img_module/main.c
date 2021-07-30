#include <stdio.h>
// #include <time.h>
#include "logger.h"
#include "adelTimer.h"
// #include "plotter.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



void
threshold (unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void 
thresholdUnrolled(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);


void 
thresholdFast(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void 
thresholdSSE2(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);


int main(int argc , char ** argv){


    // const char * filename = "imgs/lena.jpg";
    // const char * filename = "imgs/bigCat.jpg";
    char const * filename = "imgs/hires.jpg";
    char const *saveTo1 = "imgs/Thesholded-seq.jpg";
    char const *saveTo2 = "imgs/Thesholded-unroll.jpg";
    char const *saveTo3 = "imgs/Thesholded-fast.jpg";
    char const *saveTo4 = "imgs/Thesholded-SSE2.jpg";

    
    
    int width,height,channels;
    unsigned char thresholdVal = 100;

    unsigned char *data = stbi_load(filename,&width, &height, &channels, 0);
    unsigned char * dstData1 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData2 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData3 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData4 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);

     
    // double timeMeasures[4] = {0.0};

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n",filename,width,height,channels);
    logEndLine();
    startTimer();    
    threshold(data,dstData1,width,height,channels,thresholdVal);
    int msec = endTimer();
    // timeMeasures[0] = msec;
    logInfo("sequential thresholding took, %d ms\n", msec);


    startTimer();    
    thresholdUnrolled(data,dstData2,width,height,channels,thresholdVal);
    msec = endTimer();
    // timeMeasures[1] = msec;
    logInfo("unrolled thresholding took, %d ms\n", msec);



    startTimer();    
    thresholdFast(data,dstData3,width,height,channels,thresholdVal);
    msec = endTimer();
    // timeMeasures[2] = msec;
    logInfo("Fast thresholding took, %d ms\n", msec);


    startTimer();    
    thresholdSSE2(data,dstData4,width,height,channels,thresholdVal);
    msec = endTimer();
    // timeMeasures[3] = msec;
    logInfo("SSE2 thresholding took, %d ms\n", msec);


    // plot1d(timeMeasures,4,"threshold BenchMarking","different versions","time [ms]","linespoints");

    stbi_write_png(saveTo1, width,height,channels, (const void *)dstData1, width * channels);
    stbi_write_png(saveTo2, width,height,channels, (const void *)dstData2, width * channels);
    stbi_write_png(saveTo3, width,height,channels, (const void *)dstData3, width * channels);
    stbi_write_png(saveTo4, width,height,channels, (const void *)dstData4, width * channels);
     

    stbi_image_free(data);
    free(dstData1);
    free(dstData2);
    free(dstData3);
    free(dstData4);

    return 0;
}





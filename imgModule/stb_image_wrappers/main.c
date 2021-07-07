#include <stdio.h>
#include <time.h>


#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

clock_t startTime;


void
threshold (unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void 
thresholdUnrolled(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);

void startTimer(void);
int endTimer(void);


int main(int argc , char ** argv){

    // const char * filename = "imgs/lena.jpg";
    // char const *saveTo = "imgs/lenaThesholded.jpg";
    // const char * filename = "imgs/bigCat.jpg";
    // char const *saveTo = "imgs/bigCatThesholded.jpg";
    char const * filename = "imgs/hires.jpg";
    char const *saveTo1 = "imgs/hiresThesholded-seq.jpg";
    char const *saveTo2 = "imgs/hiresThesholded-unroll.jpg";

    
    
    int width,height,channels;

    unsigned char *data = stbi_load(filename,&width, &height, &channels, 0);
    unsigned char * dstData1 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData2 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n",filename,width,height,channels);

    startTimer();    
    threshold(data,dstData1,width,height,channels,150);
    int msec = endTimer();
    printf("sequential thresholding took,%d milliseconds\n", msec);

    startTimer();    
    thresholdUnrolled(data,dstData2,width,height,channels,150);
    msec = endTimer();
    printf("unrolled thresholding took,%d milliseconds\n", msec);


    stbi_write_png(saveTo1, width,height,channels, (const void *)dstData1, width * channels);
    stbi_write_png(saveTo2, width,height,channels, (const void *)dstData2, width * channels);
     

    stbi_image_free(data);
    free(dstData1);
    free(dstData2);

    return 0;
}





inline void startTimer(void){
    startTime = clock();
}


inline int endTimer(void){
    clock_t diff = clock() - startTime;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    return msec;
}




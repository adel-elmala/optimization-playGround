#include <stdio.h>
#include <unistd.h>
// #include <time.h>

#include "logger.h"
#include "adelTimer.h"
// #include "plotter.h"
#include "imgProcessingModule.h"



#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"




int main(int argc , char ** argv){


    // const char * filename = "imgs/lena.jpg";
    const char * filename = "imgs/img1.jpg";
    const char * filename2 = "imgs/img3.jpg";
    // char const * filename = "imgs/hires.jpg";
    char const * saveTo1  = "imgs/Thesholded-seq.jpg";
    char const * saveTo2  = "imgs/Thesholded-unroll.jpg";
    char const * saveTo3  = "imgs/Thesholded-fast.jpg";
    char const * saveTo4  = "imgs/Thesholded-SSE2.jpg";
    char const * saveTo5  = "imgs/Thesholded-parallel.jpg";
    // -----------------------------
    char const * saveTo6  = "imgs/cropped.jpg";
    char const * saveTo7  = "imgs/croppedSlow.jpg";
    char const * saveTo8  = "imgs/blend.jpg";

    
    
    int width,height,channels;
    int width2,height2,channels2;
    unsigned char thresholdVal = 255;

    unsigned char *data = stbi_load(filename,&width, &height, &channels, 0);
    unsigned char *data2 = stbi_load(filename2,&width2, &height2, &channels2, 0);
    unsigned char * dstData1 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData2 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData3 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData4 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char * dstData5 = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);


    testFeatureSupport();
    logEndLine();


    // double timeMeasures[5] = {0.0};

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n",filename,width,height,channels);
    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n",filename2,width2,height2,channels2);
    // logEndLine();
    logStartLine("testing thresholdXXXX()");
    startTimer();    
    threshold(data,dstData1,width,height,channels,thresholdVal);
    int uSec = endTimer();
    // timeMeasures[0] = uSec;
    logInfo("sequential thresholding took, %0.3f mSec\n", (float) uSec/1000.0f);


    startTimer();    
    thresholdUnrolled(data,dstData2,width,height,channels,thresholdVal);
    uSec = endTimer();
    // timeMeasures[1] = uSec;
    logInfo("unrolled thresholding took, %0.3f mSec\n", (float) uSec/1000.0f);




    startTimer();    
    thresholdFast(data,dstData3,width,height,channels,thresholdVal);
    uSec = endTimer();
    // timeMeasures[2] = uSec;
    logInfo("Fast thresholding took, %0.3f mSec\n", (float) uSec/1000.0f);


    startTimer();    
    thresholdSSE2(data,dstData4,width,height,channels,thresholdVal);
    uSec = endTimer();
    // timeMeasures[3] = uSec;
    logInfo("SSE2 thresholding took, %0.3f mSec\n", (float) uSec/1000.0f);


    startTimer();    
    thresholdParallel(data,dstData5,width,height,channels,thresholdVal);
    uSec = endTimer();
    // timeMeasures[4] = uSec;
    logInfo("parallel thresholding took, %0.3f mSec\n", (float) uSec/1000.0f);
    
    
    logStartLine("testing 'cropXXXX()'");


    startTimer();    
    unsigned char * cropped = crop(data,width,height,channels,100,300,150,350);
    uSec = endTimer();
    // timeMeasures[4] = uSec;
    logInfo("'crop()' took, %0.3f mSec\n", (float) uSec/1000.0f);
    


    startTimer();    
    unsigned char * croppedSlow = cropSlow(data,width,height,channels,100,300,150,350);
    // unsigned char * croppedSlow = cropSlow(data,width,height,channels,0,width,0,height);
    uSec = endTimer();
    // timeMeasures[4] = uSec;
    logInfo("'cropSlow()' took, %0.3f mSec\n", (float) uSec/1000.0f);

    logStartLine("testing 'alphaBlendXXXX()");

    startTimer();    
    unsigned char * blended = alphaBlendSSE(data,data2,width,height,channels,125);
    uSec = endTimer();
    // timeMeasures[4] = uSec;
    logInfo("'alphaBlendSSE()' took, %0.3f mSec\n", (float) uSec/1000.0f);




    // plot1d(timeMeasures,4,"threshold BenchMarking","different versions","time [ms]","linespoints");

    stbi_write_png(saveTo1, width,height,channels, (const void *)dstData1, width * channels);
    stbi_write_png(saveTo2, width,height,channels, (const void *)dstData2, width * channels);
    stbi_write_png(saveTo3, width,height,channels, (const void *)dstData3, width * channels);
    stbi_write_png(saveTo4, width,height,channels, (const void *)dstData4, width * channels);
    stbi_write_png(saveTo5, width,height,channels, (const void *)dstData5, width * channels);
    stbi_write_png(saveTo6, 200,200,channels, (const void *)cropped, 200 * channels);
    stbi_write_png(saveTo7, 200,200,channels, (const void *)croppedSlow, 200 * channels);
    // stbi_write_png(saveTo7, width,height,channels, (const void *)croppedSlow, width * channels);
    stbi_write_png(saveTo8, width,height,channels, (const void *)blended, width * channels);
     

    stbi_image_free(data);
    free(dstData1);
    free(dstData2);
    free(dstData3);
    free(dstData4);
    free(dstData5);
    free(cropped);
    free(croppedSlow);
    free(blended);

    return 0;
}





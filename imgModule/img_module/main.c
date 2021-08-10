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

int main(int argc, char **argv)
{

    // const char * filename = "imgs/lena.jpg";
    const char *filename = "imgs/img1.jpg";
    const char *filename2 = "imgs/img3.jpg";
    // char const * filename3 = "imgs/bigCat.jpg";
    char const *filename3 = "imgs/snp4.png";
    char const *filename4 = "imgs/mri.png";
    char const *filename5 = "imgs/chess.png";
    // char const * filename = "imgs/hires.jpg";

    // -----------------------------
    // -----------------------------

    char const *saveTo1 = "imgs/Thesholded-seq.jpg";
    char const *saveTo2 = "imgs/Thesholded-unroll.jpg";
    char const *saveTo3 = "imgs/Thesholded-fast.jpg";
    char const *saveTo4 = "imgs/Thesholded-SSE2.jpg";
    char const *saveTo5 = "imgs/Thesholded-parallel.jpg";
    // -----------------------------
    char const *saveTo6 = "imgs/cropped.jpg";
    char const *saveTo7 = "imgs/croppedSlow.jpg";
    // -----------------------------
    char const *saveTo8 = "imgs/blend.jpg";
    // -----------------------------
    char const *saveTo9 = "imgs/negative.jpg";
    char const *saveTo12 = "imgs/negativeAVX.jpg";
    // -----------------------------
    char const *saveTo10 = "imgs/blurred.jpg";
    char const *saveTo11 = "imgs/blurredGuassian.jpg";
    // -----------------------------
    char const *saveTo13 = "imgs/sobelX.png";
    // char const *saveTo14 = "imgs/template.jpg";

    // -----------------------------------------------------------
    int width, height, channels;
    int width2, height2, channels2;
    int width3, height3, channels3 = 1;
    int width4, height4, channels4;
    int width5, height5, channels5;
    // int width6, height6, channels6;
    unsigned char thresholdVal = 125;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);
    unsigned char *data2 = stbi_load(filename2, &width2, &height2, &channels2, 0);
    unsigned char *data3 = stbi_load(filename3, &width3, &height3, &channels3, 1);
    unsigned char *data4 = stbi_load(filename4, &width4, &height4, &channels4, 0);
    unsigned char *data5 = stbi_load(filename5, &width5, &height5, &channels5, 1);
    // unsigned char *data6 = stbi_load(filename5, &width6, &height6, &channels6, 1);
    // -----------------------------------------------------------

    testFeatureSupport();
    logEndLine();


    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename, width, height, channels);
    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename2, width2, height2, channels2);
    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename3, width3, height3, channels3);
    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename4, width4, height4, channels4);
    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename5, width5, height5, channels5);
    // printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename5, width6, height6, channels6);

    // -----------------------------------------------------------



    unsigned char *dstData1 = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstData2 = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstData3 = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstData4 = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstData5 = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);

    double timeMeasures[5] = {0.0};
    // logEndLine();
    logStartLine("testing thresholdXXXX()");
    startTimer();
    threshold(data, dstData1, width, height, channels, thresholdVal);
    int uSec = endTimer();
    timeMeasures[0] = uSec;
    logInfo("sequential thresholding took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    thresholdUnrolled(data, dstData2, width, height, channels, thresholdVal);
    uSec = endTimer();
    timeMeasures[1] = uSec;
    logInfo("unrolled thresholding took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    thresholdFast(data, dstData3, width, height, channels, thresholdVal);
    uSec = endTimer();
    timeMeasures[2] = uSec;
    logInfo("Fast thresholding took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    thresholdSSE2(data, dstData4, width, height, channels, thresholdVal);
    uSec = endTimer();
    timeMeasures[3] = uSec;
    logInfo("SSE2 thresholding took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    thresholdParallel(data, dstData5, width, height, channels, thresholdVal);
    uSec = endTimer();
    timeMeasures[4] = uSec;
    logInfo("parallel thresholding took, %0.3f mSec\n", (float)uSec / 1000.0f);
    // -----------------------------------------------------------

    logStartLine("testing 'cropXXXX()'");
    startTimer();
    unsigned char *cropped = crop(data, width, height, channels, 250, 300, 250, 300);
    uSec = endTimer();
    logInfo("'crop()' took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    unsigned char *croppedSlow = cropSlow(data, width, height, channels, 100, 300, 250, 450);
    // unsigned char * croppedSlow = cropSlow(data,width,height,channels,0,width,0,height);
    uSec = endTimer();
    logInfo("'cropSlow()' took, %0.3f mSec\n", (float)uSec / 1000.0f);
    // -----------------------------------------------------------

    logStartLine("testing 'alphaBlendXXXX()");

    startTimer();
    unsigned char *blended = alphaBlendSSE(data, data2, width, height, channels, 200);
    uSec = endTimer();
    logInfo("'alphaBlendSSE()' took, %0.3f mSec\n", (float)uSec / 1000.0f);
    // -----------------------------------------------------------

    logStartLine("testing 'negative()");

    startTimer();
    unsigned char *neg = negative(data4, width4, height4, channels4);
    uSec = endTimer();
    logInfo("'negative()' took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    unsigned char *negSSE2 = negativeSSE2(data4, width4, height4, channels4);
    uSec = endTimer();
    logInfo("'negativeAVX()' took, %0.3f mSec\n", (float)uSec / 1000.0f);
    // -----------------------------------------------------------

    logStartLine("testing 'blurXXX()");
    startTimer();
    unsigned char *blurred = blur(data3, width3, height3);
    uSec = endTimer();
    logInfo("'blur()' took, %0.3f mSec\n", (float)uSec / 1000.0f);

    startTimer();
    unsigned char *blurredG = guassianBlur(data3, width3, height3);
    uSec = endTimer();
    logInfo("'guassianBlur()' took, %0.3f mSec\n", (float)uSec / 1000.0f);
    // -----------------------------------------------------------

    logStartLine("testing 'sobel()");
    startTimer();
    unsigned char *sobellll = sobelX(data5, width5, height5);
    uSec = endTimer();
    logInfo("'sobelX()' took, %0.3f mSec\n", (float)uSec / 1000.0f);
 
    // -----------------------------------------------------------

    // plot1d(timeMeasures,4,"threshold BenchMarking","different versions","time [us]","linespoints");
    // -----------------------------------------------------------
    stbi_write_png(saveTo1, width, height, channels, (const void *)dstData1, width * channels);
    stbi_write_png(saveTo2, width, height, channels, (const void *)dstData2, width * channels);
    stbi_write_png(saveTo3, width, height, channels, (const void *)dstData3, width * channels);
    stbi_write_png(saveTo4, width, height, channels, (const void *)dstData4, width * channels);
    stbi_write_png(saveTo5, width, height, channels, (const void *)dstData5, width * channels);
    // -----------------------------------------------------------

    stbi_write_png(saveTo6, 50, 50, channels, (const void *)cropped, 50 * channels);
    stbi_write_png(saveTo7, 200, 200, channels, (const void *)croppedSlow, 200 * channels);
    // stbi_write_png(saveTo7, width,height,channels, (const void *)croppedSlow, width * channels);
    // -----------------------------------------------------------
    stbi_write_png(saveTo8, width, height, channels, (const void *)blended, width * channels);
    // -----------------------------------------------------------
    stbi_write_png(saveTo9, width4, height4, channels4, (const void *)neg, width4 * channels4);
    stbi_write_png(saveTo12, width4, height4, channels4, (const void *)negSSE2, width4 * channels4);
    // -----------------------------------------------------------
    stbi_write_png(saveTo10, width3 - 2, height3 - 2, 1, (const void *)blurred, (width3 - 2) * 1);
    // stbi_write_png(saveTo11, width3-4, height3-4, 1, (const void *)blurredG, (width3-4) * 1);
    stbi_write_png(saveTo11, width3 - 2, height3 - 2, 1, (const void *)blurredG, (width3 - 2) * 1);
    // -----------------------------------------------------------
    stbi_write_png(saveTo13, width5 - 2, height5 - 2, 1, (const void *)sobellll, (width5 - 2) * 1);
    // stbi_write_png(saveTo13, width5 - 2, height5 - 2, 1, (const void *)sobel, (width5 - 2) * 1);
    // stbi_write_png(saveTo13, width6, height6, 1, (const void *)templateM, width6 * 1);

    // -----------------------------------------------------------
    stbi_image_free(data);
    stbi_image_free(data2);
    stbi_image_free(data3);
    stbi_image_free(data4);
    stbi_image_free(data5);
    // stbi_image_free(data6);
    free(dstData1);
    free(dstData2);
    free(dstData3);
    free(dstData4);
    free(dstData5);
    free(cropped);
    free(croppedSlow);
    free(blended);
    free(neg);
    free(negSSE2);
    free(blurred);
    free(blurredG);
    free(sobellll);
    // free(templateM);

    return 0;
}

#ifndef IMGPROCESSINGMODULE_H
#define IMGPROCESSINGMODULE_H

void testFeatureSupport(void);

void threshold(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue);

void thresholdUnrolled(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue);

void thresholdFast(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue);

void thresholdSSE2(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue);

void thresholdParallel(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue);

// crops an img at colomns[x1:x2] and rows[y1:y2]
unsigned char *crop(unsigned char *srcData, int srcWidth, int srcHeight, int channels, int x1, int x2, int y1, int y2);
unsigned char *cropSlow(unsigned char *srcData, int srcWidth, int srcHeight, int channels, int x1, int x2, int y1, int y2);

// blend 2 images : alpha * img1 + (1 - alpha) * img2
unsigned char *alphaBlendSSE(unsigned char *srcData1, unsigned char *srcData2, int width, int height, int channels, unsigned char alpha);

//  gets the negative image of the input image
unsigned char *negative(unsigned char *srcData, int width, int height, int channels);
unsigned char *negativeSSE2(unsigned char *srcData, int width, int height, int channels);

// works only on 1 channel images
unsigned char *blur(unsigned char *srcData, int width, int height);
unsigned char *guassianBlur(unsigned char *srcData, int width, int height);

// unsigned char *templateMatch(unsigned char *srcData, unsigned char *template, int srcWidth, int srcHeight, int templWidth, int templHeight);
unsigned char *sobelX(unsigned char *srcData, int width, int height);
unsigned char *sobelY(unsigned char *srcData, int width, int height);

void correlate(unsigned char *srcData, unsigned char *dstData, unsigned char *mask, int width, int height, int maskWidth, int divBy);
void correlateSigned(unsigned char *srcData, unsigned char *dstData, double *mask, int width, int height, int maskWidth, int divBy);

unsigned char *EdgeDetection(unsigned char *srcData, int width, int height);
unsigned char *imgMultiply(unsigned char *img1, unsigned char *img2, int width, int height);
unsigned char *imgMultiplySSE(unsigned char *img1, unsigned char *img2, int width, int height);
unsigned char *imgAdd(unsigned char *img1, unsigned char *img2, int width, int height);
unsigned char *imgSqrt(unsigned char *srcData, int width, int height);
unsigned char *imgAtan2(unsigned char *srcData1, unsigned char *srcData2, int width, int height);

unsigned char *imgBinary(unsigned char *srcData, int width, int height, unsigned char thresholdValue);

unsigned char *canny(unsigned char *srcData, int width, int height);

unsigned char *non_maximum_Suppression(unsigned char *grad_mag, unsigned char *grad_dir, int width, int height);
float getQuadrant(float x);
unsigned char *hysteresis(unsigned char *srcData, int width, int height, unsigned char thresholdValue);



#endif
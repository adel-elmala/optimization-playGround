#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <assert.h>
#include <sys/sysinfo.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#include "imgProcessingModule.h"

typedef struct
{
    unsigned char grey;
    unsigned char alpha;
} pixel2;

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel3;

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel4;

#define _mm_cmpge_epu8(a, b) \
    _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

#define _mm_cmple_epu8(a, b) _mm_cmpge_epu8(b, a)

#define _mm_cmpgt_epu8(a, b) \
    _mm_xor_si128(_mm_cmple_epu8(a, b), _mm_set1_epi8(-1))

#define _mm_cmplt_epu8(a, b) _mm_cmpgt_epu8(b, a)

void threshold(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue)
{
    // loop rows
    for (int i = 0; i < width; ++i)
    {

        int rowStride = i * height * channels;

        // loop cols
        for (int j = 0; j < height * channels; ++j)
        {

            unsigned char pVal = srcData[rowStride + j];
            dstData[rowStride + j] = (pVal < thresholdValue) ? 0 : pVal;
        }
    }
}

void thresholdUnrolled(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue)
{
    // loop rows
    for (int i = 0; i < width; ++i)
    {

        int rowStride = i * height * channels;

        // loop cols
        for (int j = 0; j < height * channels; j += 4)
        {

            unsigned char pVal1 = srcData[rowStride + j];
            unsigned char pVal2 = srcData[rowStride + j + 1];
            unsigned char pVal3 = srcData[rowStride + j + 2];
            unsigned char pVal4 = srcData[rowStride + j + 3];

            dstData[rowStride + j] = (pVal1 < thresholdValue) ? 0 : pVal1;
            dstData[rowStride + j + 1] = (pVal2 < thresholdValue) ? 0 : pVal2;
            dstData[rowStride + j + 2] = (pVal3 < thresholdValue) ? 0 : pVal3;
            dstData[rowStride + j + 3] = (pVal4 < thresholdValue) ? 0 : pVal4;
        }
    }
}

void thresholdFast(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue)
{
    // loop rows
    unsigned char *end = srcData + (width * height * channels);
    for (unsigned char *i = srcData, *j = dstData; i < end; i += 4, j += 4)
    {
        unsigned char pVal = *i;
        unsigned char pVal1 = *(i + 1);
        unsigned char pVal2 = *(i + 2);
        unsigned char pVal3 = *(i + 3);
        *j = (pVal < thresholdValue) ? 0 : pVal;
        *(j + 1) = (pVal1 < thresholdValue) ? 0 : pVal1;
        *(j + 2) = (pVal2 < thresholdValue) ? 0 : pVal2;
        *(j + 3) = (pVal3 < thresholdValue) ? 0 : pVal3;
    }
}

void thresholdSSE2(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue)
{
    const int regWidth = 16;
    long int nBytes = width * height * channels;
    long int nChunks = nBytes / regWidth;
    assert(nChunks > 1);

    int residual = nBytes - (nChunks * regWidth);

    __m128i *end = ((__m128i *)(srcData)) + nChunks;

    // fill reg with threashold Value
    __m128i thresholdReg = _mm_set1_epi8(thresholdValue);

    for (__m128i *i = (__m128i *)srcData, *j = (__m128i *)dstData; i < end; ++i, ++j)
    {

        // load 16 bytes at a time
        __m128i srcReg = _mm_loadu_si128(i);

        // compare byte-byte  srcReg[0..15] > thresholdReg[0..15]
        __m128i resultMask = _mm_cmpgt_epu8(srcReg, thresholdReg); // 0xFF if srcReg[] > thresholdReg[] and 0 otherwise
        // and mask with srcReg
        __m128i dstReg = _mm_and_si128(srcReg, resultMask);
        // store Result back to memory
        _mm_storeu_si128(j, dstReg);
    }

    if (residual > 0)
    {
        unsigned char *start = srcData + (nChunks * regWidth);
        unsigned char *end = srcData + (width * height * channels);
        for (unsigned char *i = start, *j = dstData; i < end; ++i, ++j)
        {

            unsigned char pVal = *i;
            *j = (pVal < thresholdValue) ? 0 : pVal;
        }
    }
}

// TODO: divide work bt/w threads :DONE:
typedef struct
{
    /* data */
    unsigned char *srcData;
    unsigned char *dstData;
    int width;
    int height;
    int channels;
    unsigned char thresholdValue;
} kernalArgs_t;

void *thresholdSSEKernal(void *args)
{
    kernalArgs_t *kArgs = (kernalArgs_t *)args;

    const int regWidth = 16;
    long int nBytes = (kArgs->width) * (kArgs->height) * (kArgs->channels);
    long int nChunks = nBytes / regWidth;
    assert(nChunks > 1);

    int residual = nBytes - (nChunks * regWidth);

    __m128i *end = ((__m128i *)(kArgs->srcData)) + nChunks;

    // fill reg with threashold Value
    __m128i thresholdReg = _mm_set1_epi8(kArgs->thresholdValue);

    for (__m128i *i = (__m128i *)(kArgs->srcData), *j = (__m128i *)(kArgs->dstData); i < end; ++i, ++j)
    {

        // load 16 bytes at a time
        __m128i srcReg = _mm_loadu_si128(i);

        // compare byte-byte  srcReg[0..15] > thresholdReg[0..15]
        __m128i resultMask = _mm_cmpgt_epu8(srcReg, thresholdReg); // 0xFF if srcReg[] > thresholdReg[] and 0 otherwise
        // and mask with srcReg
        __m128i dstReg = _mm_and_si128(srcReg, resultMask);
        // store Result back to memory
        _mm_storeu_si128(j, dstReg);
    }

    if (residual > 0)
    {
        unsigned char *start = (kArgs->srcData) + (nChunks * regWidth);
        unsigned char *end = (kArgs->srcData) + ((kArgs->width) * (kArgs->height) * (kArgs->channels));
        for (unsigned char *i = start, *j = (kArgs->dstData); i < end; ++i, ++j)
        {

            unsigned char pVal = *i;
            *j = (pVal < (kArgs->thresholdValue)) ? 0 : pVal;
        }
    }
}

void thresholdParallel(unsigned char *srcData, unsigned char *dstData, int width, int height, int channels, unsigned char thresholdValue)
{
    // get how many proccessors available (slow)
    int nprocs = get_nprocs();
    // devide the img into equal size horizontal segments
    int segmentHeight = height / nprocs;
    kernalArgs_t *kargs = (kernalArgs_t *)malloc(sizeof(kernalArgs_t) * nprocs);
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * nprocs);

    for (int i = 0; i < nprocs; ++i)
    {
        kargs[i].srcData = srcData + (width * (segmentHeight * i) * channels);
        kargs[i].dstData = dstData + (width * (segmentHeight * i) * channels);
        kargs[i].width = width;
        kargs[i].height = segmentHeight;
        kargs[i].channels = channels;
        kargs[i].thresholdValue = thresholdValue;
    }

    for (int i = 0; i < nprocs; ++i)
        pthread_create(tid + i, NULL, thresholdSSEKernal, (void *)(kargs + i));

    for (int i = 0; i < nprocs; ++i)
        pthread_join(tid[i], NULL);

    free(kargs);
    free(tid);
}

// crops an img at colomns[x1:x2] and rows[y1:y2]
unsigned char *crop(unsigned char *srcData, int srcWidth, int srcHeight, int channels, int x1, int x2, int y1, int y2)
{
    assert(x1 >= 0);
    assert(y1 >= 0);
    assert(x2 >= x1);
    assert(y2 >= y1);

    int width = x2 - x1;
    int height = y2 - y1;

    // The user is responsible of freeing this block of memory
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstStart = dstData;
    unsigned char *srcStart = srcData + (y1 * srcWidth * channels) + (x1 * channels);
    unsigned char *srcEnd = srcData + (y2 * srcWidth * channels) + (x1 * channels);

    // unsigned int i = 0;
    for (unsigned char *current = srcStart; (current < srcEnd);)
    {
        memcpy((void *)dstData, (const void *)current, (width * channels));
        dstData = (unsigned char *)dstData + (width * channels);
        current = (unsigned char *)current + (srcWidth * channels);
    }
    return dstStart;
}

// crops an img at colomns[x1:x2] and rows[y1:y2]
unsigned char *cropSlow(unsigned char *srcData, int srcWidth, int srcHeight, int channels, int x1, int x2, int y1, int y2)
{
    assert(x1 >= 0);
    assert(y1 >= 0);
    assert(x2 >= x1);
    assert(y2 >= y1);

    int width = x2 - x1;
    int height = y2 - y1;

    // The user is responsible of freeing this block of memory
    // printf("malloc : %lu\n",(sizeof(unsigned char) * width * height * channels));

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstStart = dstData;
    unsigned char *srcStart = srcData + (y1 * srcWidth * channels) + (x1 * channels);
    unsigned char *srcEnd = srcData + (y2 * srcWidth * channels) + (x1 * channels);

    int i = 0;
    int j = 0;
    for (unsigned char *current = srcStart; (current < srcEnd);)
    {
        *dstData = *current;
        ++dstData;
        ++i;

        if (i % (width * channels) == 0) // copied entire row
        {
            ++j;
            current = srcStart + (srcWidth * j * channels);
        }
        else
        {
            ++current;
        }
    }

    return dstStart;
}

// blend 2 images : (alpha * img1 + (255 - alpha) * img2) / 256
// divison by 256 instead of 255 for more performance
unsigned char *alphaBlendSSE(unsigned char *srcData1, unsigned char *srcData2, int width, int height, int channels, unsigned char alpha)
{

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);

    const unsigned int regSize = 16;
    const unsigned int bytesNumber = width * height * channels;
    const unsigned int chunks = bytesNumber / regSize;
    const unsigned int residiual = bytesNumber - (chunks * 16);

    __m128i zeroReg = _mm_setzero_si128();
    __m128i oneReg16 = _mm_set1_epi16(0x00FF); // filled with 16-bit 255 values
    __m128i alphaReg8 = _mm_set1_epi8(alpha);

    // reg filled with 16-bit alpha values
    __m128i alphaReg16 = _mm_unpacklo_epi8(alphaReg8, zeroReg);

    // reg filled with 16-bit (255-alpha) values
    __m128i alphaCompReg16 = _mm_sub_epi16(oneReg16, alphaReg16);

    __m128i *dstReg = (__m128i *)dstData;
    __m128i *srcEnd = ((__m128i *)srcData1) + chunks;

    for (__m128i *src1 = (__m128i *)srcData1, *src2 = (__m128i *)srcData2; src1 < srcEnd; ++src1, ++src2, ++dstReg)
    {

        // load 16 bytes from img1
        __m128i srcReg1 = _mm_loadu_si128((__m128i const *)src1);

        // load 16 bytes from img2
        __m128i srcReg2 = _mm_loadu_si128((__m128i const *)src2);

        // unpack low-8 Bytes to be extended to 16-bit each with zeros
        __m128i srcReg1Lo = _mm_unpacklo_epi8(srcReg1, zeroReg);
        // unpack high-8 Bytes to be extended to 16-bit each with zeros
        __m128i srcReg1Hi = _mm_unpackhi_epi8(srcReg1, zeroReg);

        // unpack low-8 Bytes to be extended to 16-bit each with zeros
        __m128i srcReg2Lo = _mm_unpacklo_epi8(srcReg2, zeroReg);
        // unpack high-8 Bytes to be extended to 16-bit each with zeros
        __m128i srcReg2Hi = _mm_unpackhi_epi8(srcReg2, zeroReg);

        srcReg1Lo = _mm_mullo_epi16(srcReg1Lo, alphaReg16);
        srcReg1Hi = _mm_mullo_epi16(srcReg1Hi, alphaReg16);

        srcReg2Lo = _mm_mullo_epi16(srcReg2Lo, alphaCompReg16);
        srcReg2Hi = _mm_mullo_epi16(srcReg2Hi, alphaCompReg16);

        __m128i rsltRegLo = _mm_add_epi16(srcReg1Lo, srcReg2Lo);
        __m128i rsltRegHi = _mm_add_epi16(srcReg1Hi, srcReg2Hi);
        rsltRegLo = _mm_srli_epi16(rsltRegLo, 8);
        rsltRegHi = _mm_srli_epi16(rsltRegHi, 8);

        _mm_store_si128(dstReg, _mm_packus_epi16(rsltRegLo, rsltRegHi));
    }

    return dstData;
}

void testFeatureSupport(void)
{
    __builtin_cpu_init();
    // deteced if intel
    if (__builtin_cpu_is("intel"))
    {
        printf("your cpu is an INTEL!\n");

        if (__builtin_cpu_supports("sse"))
            printf("your cpu supports SSE!\n");
        else
            printf("your cpu does NOT supports SSE!\n");
    }

    else
        printf("your cpu is NOT an INTEL :< !\n");
}

//  gets the negative image of the input image
unsigned char *negative(unsigned char *srcData, int width, int height, int channels)
{
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstStart = dstData;
    unsigned char *srcEnd = srcData + (width * height * channels);

    for (unsigned char *current = srcData; current < srcEnd; ++current, ++dstData)
        *dstData = 255 - (*current);

    return dstStart;
}

//  gets the negative image of the input image
unsigned char *negativeSSE2(unsigned char *srcData, int width, int height, int channels)
{
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
    unsigned char *dstStart = dstData;
    const unsigned int regSize = 16;
    const unsigned int nBytes = (width * height * channels);
    const unsigned int chunks = nBytes / regSize;
    unsigned int residuial = nBytes - (chunks * regSize);
    __m128i *srcEnd = ((__m128i *)srcData) + chunks;

    __m128i high = _mm_set1_epi8(0xff);

    for (__m128i *current = (__m128i *)srcData, *dstReg = (__m128i *)dstData; current < srcEnd; ++current, ++dstReg)
    {
        // load 32 bytes at once
        __m128i srcReg = _mm_loadu_si128((__m128i const *)current);

        __m128i result = _mm_subs_epu8(high, srcReg);
        _mm_storeu_si128(dstReg, result);
    }

    if (residuial != 0)
    {
        unsigned char *src = (unsigned char *)srcEnd;
        unsigned char *dst = (unsigned char *)((__m128i *)dstData + chunks);

        for (; residuial != 0; --residuial)
            *dst = 255 - *src;
    }

    return dstStart;
}

unsigned char *blur(unsigned char *srcData, int width, int height)
{
    // unsigned char avgMask[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // check is same image :DONE:
    unsigned char avgMask[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int maskWidth = 3;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    unsigned char *srcEnd = srcData + (width * height);
    correlate(srcData, dstData, avgMask, width, height, maskWidth, 9);

    return dstStart;
}

unsigned char *guassianBlur(unsigned char *srcData, int width, int height)
{
    // unsigned char avgMask[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // check is same image :DONE:
    unsigned char avgMask[9] = {1, 2, 1,
                                2, 4, 2,
                                1, 2, 1};

    // unsigned char avgMask[25] = {1, 4, 7, 4, 1,
    //                              4, 16, 26, 16, 4,
    //                              7, 26, 41, 26, 7,
    //                              4, 16, 26, 16, 4,
    //                              1, 4, 7, 4, 1};
    int maskWidth = 3;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    unsigned char *srcEnd = srcData + (width * height);
    correlate(srcData, dstData, avgMask, width, height, maskWidth, 16);

    return dstStart;
}

unsigned char *guassianBlur5X5(unsigned char *srcData, int width, int height)
{
    // unsigned char avgMask[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // check is same image :DONE:
    unsigned char avgMask[25] = {2, 4, 5, 4, 2,
                                 4, 9, 12, 9, 4,
                                 5, 12, 15, 12, 5,
                                 4, 9, 12, 9, 4,
                                 2, 4, 5, 4, 2};

    // unsigned char avgMask[25] = {1, 4, 7, 4, 1,
    //                              4, 16, 26, 16, 4,
    //                              7, 26, 41, 26, 7,
    //                              4, 16, 26, 16, 4,
    //                              1, 4, 7, 4, 1};
    int maskWidth = 5;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    unsigned char *srcEnd = srcData + (width * height);
    correlate(srcData, dstData, avgMask, width, height, maskWidth, 159);

    return dstStart;
}

void correlateSigned(unsigned char *srcData, unsigned char *dstData, double *mask, int width, int height, int maskWidth, int divBy)
{

    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    // unsigned char *srcStart = srcData + (width * channels * floatingEdges) + (channels * floatingEdges);
    int rowEnd = height - floatingEdges;
    int colEnd = width - floatingEdges;
    // for each row
    for (int i = floatingEdges, i2 = 0; i < rowEnd; ++i, ++i2)
    {
        // for each col
        for (int j = floatingEdges, j2 = 0; j < colEnd; ++j, ++j2)
        {
            double sum = 0;
            // for mask rows
            for (int k = 0; k < maskWidth; ++k)
            {
                // for mask cols
                for (int l = 0; l < maskWidth; ++l)
                {
                    int relativeRow = -floatingEdges + k;
                    int relativeCol = -floatingEdges + l;
                    sum += mask[(k * maskWidth) + l] * ((double)srcData[((i + relativeRow) * width) + (j + relativeCol)]);
                }
            }
            sum /= (double)divBy;
            if (sum > 255.0)
                sum = 255.0;
            dstData[(i2 * (width - padding)) + j2] = (unsigned char)sum;
        }
    }
}

void correlate(unsigned char *srcData, unsigned char *dstData, unsigned char *mask, int width, int height, int maskWidth, int divBy)
{
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    // unsigned char *srcStart = srcData + (width * channels * floatingEdges) + (channels * floatingEdges);
    int rowEnd = height - floatingEdges;
    int colEnd = width - floatingEdges;
    // for each row
    for (int i = floatingEdges, i2 = 0; i < rowEnd; ++i, ++i2)
    {
        // for each col
        for (int j = floatingEdges, j2 = 0; j < colEnd; ++j, ++j2)
        {
            double sum = 0;
            // for mask rows
            for (int k = 0; k < maskWidth; ++k)
            {
                // for mask cols
                for (int l = 0; l < maskWidth; ++l)
                {
                    int relativeRow = -floatingEdges + k;
                    int relativeCol = -floatingEdges + l;
                    sum += mask[(k * maskWidth) + l] * srcData[((i + relativeRow) * width) + (j + relativeCol)];
                }
            }
            sum /= divBy;
            if (sum > 255.0)
                sum = 255.0;
            dstData[(i2 * (width - padding)) + j2] = (unsigned char)sum;
        }
    }
}

// // works only with single-channel imgs
// unsigned char *templateMatch(unsigned char *srcData, unsigned char *template, int srcWidth, int srcHeight, int templWidth, int templHeight)
// {
//     unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * srcWidth * srcHeight);
//     correlate(srcData, dstData, template, srcWidth, srcHeight, templWidth, 1);
//     return dstData;
// }

unsigned char *sobelX(unsigned char *srcData, int width, int height)
{
    // unsigned char sobel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // check is same image :DONE:

    // :FIXME: correlation to accept negative values
    double sobel[9] = {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1};
    int maskWidth = 3;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    // unsigned char *srcEnd = srcData + (width * height);
    correlateSigned(srcData, dstData, sobel, width, height, maskWidth, 8);

    return dstStart;
}

unsigned char *sobelY(unsigned char *srcData, int width, int height)
{
    // unsigned char sobel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // check is same image :DONE:

    // :FIXME: correlation to accept negative values
    double sobel[9] = {1, 2, 1,
                       0, 0, 0,
                       -1, -2, -1};
    int maskWidth = 3;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    // unsigned char *srcEnd = srcData + (width * height);
    correlateSigned(srcData, dstData, sobel, width, height, maskWidth, 8);

    return dstStart;
}

unsigned char *EdgeDetection(unsigned char *srcData, int width, int height)
{
    unsigned char *blurred = guassianBlur5X5(srcData, width, height);

    unsigned char *dx = sobelX(blurred, width - 4, height - 4);
    unsigned char *dy = sobelY(blurred, width - 4, height - 4);
    int newWidth = width - 6;
    int newHeight = height - 6;
    unsigned char *dx2 = imgMultiply(dx, dx, newWidth, newHeight);
    unsigned char *dy2 = imgMultiply(dy, dy, newWidth, newHeight);
    unsigned char *dx2_plus_dy2 = imgAdd(dx2, dy2, newWidth, newHeight);
    unsigned char *gradient_mag = imgSqrt(dx2_plus_dy2, newWidth, newHeight);
   

    // unsigned char *dx = sobelX(srcData, width, height);
    // unsigned char *dy = sobelY(srcData, width, height);
    // unsigned char *dx2 = imgMultiply(dx, dx, width - 2, height - 2);
    // unsigned char *dy2 = imgMultiply(dy, dy, width - 2, height - 2);
    // unsigned char *dx2_plus_dy2 = imgAdd(dx2, dy2, width - 2, height - 2);
    // unsigned char *gradient_mag = imgSqrt(dx2_plus_dy2, width - 2, height - 2);

    free(dx);
    free(dy);
    free(dx2);
    free(dy2);
    free(dx2_plus_dy2);
    // free(gradient_mag);
    return gradient_mag;
    
}

unsigned char *canny(unsigned char *srcData, int width, int height)
{
    unsigned char *blurred = guassianBlur5X5(srcData, width, height);

    unsigned char *dx = sobelX(blurred, width - 4, height - 4);
    unsigned char *dy = sobelY(blurred, width - 4, height - 4);
    int newWidth = width - 6;
    int newHeight = height - 6;
    unsigned char *dx2 = imgMultiply(dx, dx, newWidth, newHeight);
    unsigned char *dy2 = imgMultiply(dy, dy, newWidth, newHeight);
    unsigned char *dx2_plus_dy2 = imgAdd(dx2, dy2, newWidth, newHeight);
    unsigned char *gradient_mag = imgSqrt(dx2_plus_dy2, newWidth, newHeight);
    unsigned char *gradient_direction = imgAtan2(dy, dx, newWidth, newHeight);
    unsigned char *nonMaxThresholded = non_maximum_Suppression(gradient_mag, gradient_direction, newWidth, newHeight);
    // unsigned char *hyst = hysteresis(nonMaxThresholded, newWidth, newHeight, 70);

    free(dx);
    free(dy);
    free(dx2);
    free(dy2);
    free(dx2_plus_dy2);
    free(gradient_mag);
    free(gradient_direction);
    // free(nonMaxThresholded);
    return blurred;
}
unsigned char *non_maximum_Suppression(unsigned char *grad_mag, unsigned char *grad_dir, int width, int height)
{
    // for each pixel in grad_mag
    // check direction of grad_dir
    // and threshold if it's the max between neighbors
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    memset(dstData, 0, width * height);

    unsigned char *srcEnd = grad_mag + (width * (height - 1)) - 1;
    int counter = 0;
    int newWidth = width - 2;
    for (unsigned char *srcMag = (grad_mag + width + 1), *srcDir = (grad_dir + width + 1), *dst = (dstData + width + 1); srcMag < srcEnd; ++srcMag, ++srcDir, ++dst)
    {
        unsigned char pix = *srcMag;
        unsigned char dir = *srcDir;
        if (dir == 0)
        {
            if ((pix != 0) && ((pix) >= (*(srcMag + 1))) && ((pix) >= (*(srcMag - 1))))
                *dst = pix;
        }
        else if (dir == 45)
        {
            // unsigned char pix = *srcMag;
            if ((pix != 0) && ((pix) >= (*(srcMag - (width) + 1))) && ((pix) >= (*(srcMag + (width)-1))))
                *dst = pix;
        }
        else if (dir == 90)
        {
            // unsigned char pix = *srcMag;
            if ((pix != 0) && ((pix) >= (*(srcMag - width))) && ((pix) >= (*(srcMag + width))))
                *dst = pix;
        }
        else if (dir == 135)
        {
            if ((pix != 0) && ((pix) >= (*(srcMag - width - 1))) && ((pix) >= (*(srcMag + width + 1))))
                *dst = pix;
        }

        ++counter;
        if ((counter % newWidth) == 0)
        {
            ++srcMag;
            ++srcDir;
            ++dst;
        }
    }
    return dstData;
}

unsigned char *imgMultiplySSE(unsigned char *img1, unsigned char *img2, int width, int height)
{

    __m128i zeroReg = _mm_setzero_si128();
    int nBytes = width * height;
    unsigned char *dstData = malloc(sizeof(unsigned char) * nBytes);
    unsigned char *dstStart = dstData;

    int registerSize = 16;
    int chunks = nBytes / registerSize;
    int residual = nBytes - (chunks * registerSize);
    __m128i *src1End = (__m128i *)img1 + chunks;
    __m128i *dstReg = (__m128i *)dstData;

    for (__m128i *src1 = (__m128i *)img1, *src2 = (__m128i *)img2; src1 < src1End; ++src1, ++src2, ++dstReg)
    {

        // load 16 bytes from each img
        __m128i srcReg1 = _mm_loadu_si128((__m128i const *)src1);
        __m128i srcReg2 = _mm_loadu_si128((__m128i const *)src2);
        // unpack into 16-bit to make room for multiplication
        __m128i srcReg1_Lo16 = _mm_unpacklo_epi8(srcReg1, zeroReg);
        __m128i srcReg1_Hi16 = _mm_unpackhi_epi8(srcReg1, zeroReg);

        // unpack into 16-bit to make room for multiplication
        __m128i srcReg2_Lo16 = _mm_unpacklo_epi8(srcReg2, zeroReg);
        __m128i srcReg2_Hi16 = _mm_unpackhi_epi8(srcReg2, zeroReg);

        __m128i rslt_Lo16 = _mm_mullo_epi16(srcReg1_Lo16, srcReg2_Lo16);
        __m128i rslt_Hi16 = _mm_mullo_epi16(srcReg1_Hi16, srcReg2_Hi16);
        __m128i rslt = _mm_packus_epi16(rslt_Lo16, rslt_Hi16);
        _mm_storeu_si128(dstReg, rslt);
    }
    if (residual != 0)
    {
        // __m128i *src1End = (__m128i *)img1 + chunks;
        unsigned char *src1 = (unsigned char *)((__m128i *)img1 + chunks);
        unsigned char *src2 = (unsigned char *)((__m128i *)img2 + chunks);
        unsigned char *dst = (unsigned char *)((__m128i *)dstData + chunks);
        unsigned char *src1End = img1 + nBytes;
        while (src1 < src1End)
        {
            *dst = (*src1) * (*src2);
            ++src1;
            ++src2;
            ++dst;
        }
    }
    return dstStart;
}

unsigned char *imgAdd(unsigned char *img1, unsigned char *img2, int width, int height)
{

    int nBytes = width * height;
    unsigned char *dstData = malloc(sizeof(unsigned char) * nBytes);
    unsigned char *dstStart = dstData;
    int registerSize = 16;
    int chunks = nBytes / registerSize;
    int residual = nBytes - (chunks * registerSize);

    __m128i *src1End = (__m128i *)img1 + chunks;
    __m128i *dstReg = (__m128i *)dstData;

    for (__m128i *src1 = (__m128i *)img1, *src2 = (__m128i *)img2; src1 < src1End; ++src1, ++src2, ++dstReg)
    {

        // load 16 bytes from each img
        __m128i srcReg1 = _mm_loadu_si128((__m128i const *)src1);
        __m128i srcReg2 = _mm_loadu_si128((__m128i const *)src2);
        // add them
        __m128i rslt = _mm_adds_epu8(srcReg1, srcReg2);
        _mm_storeu_si128(dstReg, rslt);
    }
    if (residual != 0)
    {
        // __m128i *src1End = (__m128i *)img1 + chunks;
        unsigned char *src1 = (unsigned char *)((__m128i *)img1 + chunks);
        unsigned char *src2 = (unsigned char *)((__m128i *)img2 + chunks);
        unsigned char *dst = (unsigned char *)((__m128i *)dstData + chunks);
        unsigned char *src1End = img1 + nBytes;
        while (src1 < src1End)
        {
            *dst = (*src1) + (*src2);
            ++src1;
            ++src2;
            ++dst;
        }
    }
    return dstStart;
}

unsigned char *imgSqrt(unsigned char *srcData, int width, int height)
{

    int nBytes = width * height;
    unsigned char *dstData = malloc(sizeof(unsigned char) * nBytes);
    unsigned char *dstStart = dstData;
    int chuncks = nBytes / 4;
    unsigned char *srcEnd = srcData + (chuncks * 4);
    int residual = nBytes - (chuncks * 4);

    for (unsigned char *src = srcData; src < srcEnd; src += 4, dstData += 4)
    {
        float result1 = sqrtf((float)(*src));
        float result2 = sqrtf((float)(*(src + 1)));
        float result3 = sqrtf((float)(*(src + 2)));
        float result4 = sqrtf((float)(*(src + 3)));
        *dstData = result1 > 255.0f ? (unsigned char)255 : (unsigned char)result1;
        *(dstData + 1) = result2 > 255.0f ? (unsigned char)255 : (unsigned char)result2;
        *(dstData + 2) = result3 > 255.0f ? (unsigned char)255 : (unsigned char)result3;
        *(dstData + 3) = result4 > 255.0f ? (unsigned char)255 : (unsigned char)result4;
    }
    if (residual != 0)
    {
        unsigned char *sStart = srcEnd;
        unsigned char *sEnd = srcData + nBytes;
        unsigned char *dst = dstData;
        while (sStart < sEnd)
        {
            float result1 = sqrtf((float)(*sStart));
            *dst = result1 > 255.0f ? (unsigned char)255 : (unsigned char)result1;
            ++sStart;
            ++dst;
        }
    }
    return dstStart;
}


unsigned char *imgMultiply(unsigned char *img1, unsigned char *img2, int width, int height)
{

    int nBytes = width * height;
    unsigned char *dstData = malloc(sizeof(unsigned char) * nBytes);
    unsigned char *dstStart = dstData;
    int chuncks = nBytes / 4;
    unsigned char *srcEnd = img1 + (chuncks * 4);
    unsigned char *srcEnd2 = img2 + (chuncks * 4);
    int residual = nBytes - (chuncks * 4);

    for (unsigned char *src1 = img1, *src2 = img2; src1 < srcEnd; src1 += 4, src2 += 4, dstData += 4)
    {
        long int result1 = ((long int)(*src1)) * ((long int)(*src2));
        long int result2 = ((long int)(*(src1 + 1))) * ((long int)(*(src2 + 1)));
        long int result3 = ((long int)(*(src1 + 2))) * ((long int)(*(src2 + 2)));
        long int result4 = ((long int)(*(src1 + 3))) * ((long int)(*(src2 + 3)));

        *dstData = result1 > 255 ? (unsigned char)255 : (unsigned char)result1;
        *(dstData + 1) = result2 > 255 ? (unsigned char)255 : (unsigned char)result2;
        *(dstData + 2) = result3 > 255 ? (unsigned char)255 : (unsigned char)result3;
        *(dstData + 3) = result4 > 255 ? (unsigned char)255 : (unsigned char)result4;
    
    }
    if (residual != 0)
    {
        unsigned char *sStart1 = srcEnd;
        unsigned char *sStart2 = srcEnd2;
        unsigned char *sEnd = img1 + nBytes;
        unsigned char *dst = dstData;
        while (sStart1 < sEnd)
        {
        
            long int result1 = ((long int)(*sStart1)) * ((long int)(*sStart2));
            *dst = result1 > 255 ? (unsigned char)255 : (unsigned char)result1;

            ++sStart1;
            ++sStart2;
            ++dst;
        }
    }
    return dstStart;

}

unsigned char *imgAtan2(unsigned char *srcData1, unsigned char *srcData2, int width, int height)
{

    int nBytes = width * height;
    unsigned char *dstData = malloc(sizeof(unsigned char) * nBytes);
    unsigned char *dstStart = dstData;
    int chuncks = nBytes / 4;
    unsigned char *srcEnd = srcData1 + (chuncks * 4);
    unsigned char *srcEnd2 = srcData2 + (chuncks * 4);
    int residual = nBytes - (chuncks * 4);

    for (unsigned char *src1 = srcData1, *src2 = srcData2; src1 < srcEnd; src1 += 4, src2 += 4, dstData += 4)
    {
        // TODO: need to check values of atan2 if in rad or deg
        float result1 = atan2f((float)(*src1), (float)(*src2));
        float result2 = atan2f((float)(*(src1 + 1)), (float)(*(src2 + 1)));
        float result3 = atan2f((float)(*(src1 + 2)), (float)(*(src2 + 2)));
        float result4 = atan2f((float)(*(src1 + 3)), (float)(*(src2 + 3)));

        result1 = getQuadrant(result1);
        result2 = getQuadrant(result2);
        result3 = getQuadrant(result3);
        result4 = getQuadrant(result4);

        *dstData = (unsigned char)result1;
        *(dstData + 1) = (unsigned char)result2;
        *(dstData + 2) = (unsigned char)result3;
        *(dstData + 3) = (unsigned char)result4;
    }
    if (residual != 0)
    {
        unsigned char *sStart1 = srcEnd;
        unsigned char *sStart2 = srcEnd2;
        unsigned char *sEnd = srcData1 + nBytes;
        unsigned char *dst = dstData;
        while (sStart1 < sEnd)
        {
            float result1 = atan2f((float)(*sStart1), (float)(*sStart2));
            result1 = getQuadrant(result1);

            *dst = (unsigned char)result1;
            ++sStart1;
            ++sStart2;
            ++dst;
        }
    }
    return dstStart;
}

inline float getQuadrant(float x)
{
    float angle = (x / 3.14) * 180;
    float diff_from_0 = angle;
    float diff_from_45 = fabsf(angle - 45.0f);
    float diff_from_90 = fabsf(angle - 90.0f);
    float diff_from_135 = fabsf(angle - 135.0f);

    float diff_from_180 = fabsf(angle - 180.0f);
    float diff_from_225 = fabsf(angle - 225.0f);
    float diff_from_270 = fabsf(angle - 270.0f);
    float diff_from_315 = fabsf(angle - 315.0f);

    float min = fminf(diff_from_0, diff_from_45);
    min = fminf(min, diff_from_90);
    min = fminf(min, diff_from_135);
    min = fminf(min, diff_from_180);
    min = fminf(min, diff_from_225);
    min = fminf(min, diff_from_270);
    min = fminf(min, diff_from_315);
    if ((min == diff_from_0) || (min == diff_from_180))
        return 0.0f;
    else if ((min == diff_from_45) || (min == diff_from_225))
        return 45.0f;
    else if ((min == diff_from_90) || (min == diff_from_270))
        return 90.0f;
    else if ((min == diff_from_135) || (min == diff_from_315))
        return 135.0f;
}

unsigned char *imgBinary(unsigned char *srcData, int width, int height, unsigned char thresholdValue)
{
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    unsigned char *dstStart = dstData;
    // loop rows
    unsigned char *end = srcData + (width * height);
    for (unsigned char *i = srcData, *j = dstData; i < end; i += 4, j += 4)
    {
        unsigned char pVal = *i;
        unsigned char pVal1 = *(i + 1);
        unsigned char pVal2 = *(i + 2);
        unsigned char pVal3 = *(i + 3);
        *j = (pVal < thresholdValue) ? 0 : 255;
        *(j + 1) = (pVal1 < thresholdValue) ? 0 : 255;
        *(j + 2) = (pVal2 < thresholdValue) ? 0 : 255;
        *(j + 3) = (pVal3 < thresholdValue) ? 0 : 255;
    }
    return dstStart;
}

unsigned char *hysteresis(unsigned char *srcData, int width, int height, unsigned char thresholdValue)
{
    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
    memset(dstData, 0, width * height);

    unsigned char *srcEnd = srcData + (width * (height - 1)) - 1;
    int counter = 0;
    int newWidth = width - 2;

    for (unsigned char *src = (srcData + width + 1), *dst = (dstData + width + 1); src < srcEnd; ++src, ++dst)
    {
        unsigned char p11 = *(src - width - 1);
        unsigned char p12 = *(src - width);
        unsigned char p13 = *(src - width + 1);
        unsigned char p21 = *(src - 1);
        unsigned char p22 = *(src);
        unsigned char p23 = *(src + 1);
        unsigned char p31 = *(src + width - 1);
        unsigned char p32 = *(src + width);
        unsigned char p33 = *(src + width + 1);

        if ((p11 >= thresholdValue) ||
            (p12 >= thresholdValue) ||
            (p13 >= thresholdValue) ||
            (p21 >= thresholdValue) ||
            (p22 >= thresholdValue) ||
            (p23 >= thresholdValue) ||
            (p31 >= thresholdValue) ||
            (p32 >= thresholdValue) ||
            (p33 >= thresholdValue))
        {
            *dst = 255;
        }

        ++counter;
        if ((counter % newWidth) == 0)
        {
            ++src;
            ++dst;
        }
    }
    return dstData;
}

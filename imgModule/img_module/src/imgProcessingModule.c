#include <stdio.h>
#include <emmintrin.h>
#include <assert.h>
#include <sys/sysinfo.h>
#include <pthread.h>
#include <string.h>

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
    int maskWidth = 3;
    int floatingEdges = maskWidth / 2;
    int padding = floatingEdges * 2;

    unsigned char *dstData = (unsigned char *)malloc(sizeof(unsigned char) * (width - padding) * (height - padding));
    unsigned char *dstStart = dstData;
    unsigned char *srcEnd = srcData + (width * height);
    correlate(srcData, dstData, avgMask, width, height, maskWidth, 16);

    return dstStart;
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

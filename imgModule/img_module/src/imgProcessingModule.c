#include <stdio.h>
#include <emmintrin.h>
#include <assert.h>


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

#define _mm_cmpge_epu8(a, b) \
        _mm_cmpeq_epi8(_mm_max_epu8(a, b), a)

#define _mm_cmple_epu8(a, b) _mm_cmpge_epu8(b, a)

#define _mm_cmpgt_epu8(a, b) \
        _mm_xor_si128(_mm_cmple_epu8(a, b), _mm_set1_epi8(-1))

#define _mm_cmplt_epu8(a, b) _mm_cmpgt_epu8(b, a)



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



void thresholdFast(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue)
{
    // loop rows
    unsigned char* end = srcData + (width * height * channels); 
    for(unsigned char* i = srcData,*j = dstData; i < end; i+=4,j+=4){
            unsigned char pVal = *i;
            unsigned char pVal1 = *(i + 1);
            unsigned char pVal2 = *(i + 2);
            unsigned char pVal3 = *(i + 3);
            *j = (pVal < thresholdValue)? 0 : pVal ;
            *(j + 1) = (pVal1 < thresholdValue)? 0 : pVal1 ;
            *(j + 2) = (pVal2 < thresholdValue)? 0 : pVal2 ;
            *(j + 3) = (pVal3 < thresholdValue)? 0 : pVal3 ;
    }
}



void thresholdSSE2(unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue)
{
    const int regWidth = 16;
    long int nBytes = width * height * channels;
    long int nChunks = nBytes / regWidth;
    assert(nChunks > 1);

    int residual = nBytes - (nChunks * regWidth);

    __m128i* end = ((__m128i*)(srcData)) + nChunks;
    for(__m128i* i = (__m128i*)srcData,*j = (__m128i*)dstData; i < end;++i,++j)
    {

    // load 16 bytes at a time
    __m128i srcReg =  _mm_loadu_si128 (i);
    // fill reg with threashold Value
    __m128i thresholdReg = _mm_set1_epi8 (thresholdValue);
    // compare byte-byte  srcReg[0..15] > thresholdReg[0..15] 
    __m128i resultMask = _mm_cmpgt_epu8 (srcReg,thresholdReg); // 0xFF if srcReg[] > thresholdReg[] and 0 otherwise 
    // and mask with srcReg
    __m128i dstReg = _mm_and_si128(srcReg,resultMask);
    // store Result back to memory 
    _mm_storeu_si128 (j, dstReg);

    }

    if(residual > 0)
    {   
        unsigned char *start = srcData + (nChunks * regWidth); 
        unsigned char *end = srcData + (width * height * channels); 
        for(unsigned char *i = start,*j = dstData;i < end;++i,++j)
        {

            unsigned char pVal = *i;
            *j = (pVal < thresholdValue)? 0 : pVal ;

        }
    }

}


// TODO: divide work bt/w threads

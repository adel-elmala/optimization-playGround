#include <stdio.h>



#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


typedef struct{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    // unsigned char a;
} pixel3;

void
threshold (unsigned char * srcData,unsigned char * dstData,int width,int height,int channels,unsigned char thresholdValue);



int main(int argc , char ** argv){

    const char * filename = "imgs/lena.jpg";
    char const *saveTo = "imgs/lenaThesholded.jpg";
    // const char * filename = "imgs/bigCat.jpg";
    // char const *saveTo = "imgs/bigCatThesholded.jpg";
    
    int width,height,channels;

    unsigned char *data = stbi_load(filename,&width, &height, &channels, 0);
    unsigned char * dstData = (unsigned char *) malloc(sizeof(unsigned char) * width * height * channels);

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n",filename,width,height,channels);


    threshold(data,dstData,width,height,channels,0);

    stbi_write_png(saveTo, width,height,channels, (const void *)dstData, width * channels);
     

    stbi_image_free(data);
    free(dstData);

    return 0;
}









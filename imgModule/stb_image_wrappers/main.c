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


int main(int argc , char ** argv){

    const char * filename = "imgs/tomato.jpeg";
    int width,height,channels;
    
    unsigned char *data = stbi_load(filename,&width, &height, &channels, 0);


    printf("%s: width: %d px,height: %d px, channels: %d\n",filename,width,height,channels);
    printf("first pixel (r,g,b) value is: (%d,%d,%d)\n",((pixel3 *)data)[0].r,((pixel3 *)data)[0].g,((pixel3 *)data)[0].b);


    char const *saveTo = "imgs/chess22.png";
    stbi_write_png(saveTo, width,height,channels, (const void *)data, width * channels);
     

    stbi_image_free(data);

    return 0;
}









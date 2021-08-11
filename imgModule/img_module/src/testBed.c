
#include <stdio.h>
#include <unistd.h>

#include "logger.h"
#include "adelTimer.h"
#include "imgProcessingModule.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv)
{

    const char *filename = "imgs/original/chess.png";
    // const char *filename = "imgs/original/snp2.png";

    char const *saveTo = "imgs/results/data.png";
    char const *saveTo1 = "imgs/results/sobelx.png";
    char const *saveTo2 = "imgs/results/sobely.png";

    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename, width, height, channels);

    // -----------------------------------------------------------

    logStartLine("testing 'sobel()");
    startTimer();

    unsigned char * sobelx = sobelX(data, width, height);
    unsigned char * sobely = sobelY(data, width, height);
    int uSec = endTimer();
    logInfo("'sobelX()' took, %0.3f mSec\n", (float)uSec / 1000.0f);

    stbi_write_png(saveTo, width, height, 1, (const void *)data, (width)*1);
    stbi_write_png(saveTo1, width - 2, height - 2, 1, (const void *)sobelx, (width - 2) * 1);
    stbi_write_png(saveTo2, width - 2, height - 2, 1, (const void *)sobely, (width - 2) * 1);

    // -----------------------------------------------------------
    stbi_image_free(data);
    free(sobelx);
    free(sobely);

    return 0;
}

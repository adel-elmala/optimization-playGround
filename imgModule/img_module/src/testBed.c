
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

    // const char *filename = "imgs/original/chess.png"; 
    // const char *filename = "imgs/original/img1.jpg";
    const char *filename = "imgs/original/lena.jpg";

    char const *saveTo = "imgs/results/data.png";
    char const *saveTo1 = "imgs/results/EdgeDetect.png";
    // char const *saveTo2 = "imgs/results/sobely.png";
    char const *saveTo3 = "imgs/results/canny.png";

    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename, width, height, channels);

    // -----------------------------------------------------------

    logStartLine("testing 'EdgeDetection");
    startTimer();
    unsigned char *EdgeDetect = EdgeDetection(data, width, height);
    int uSec = endTimer();
    logInfo("'EdgeDetection()' took, %0.3f mSec\n", (float)uSec / 1000.0f);


    startTimer();
    unsigned char *cannyEdge = canny(data, width, height);
    uSec = endTimer();
    logInfo("'canny()' took, %0.3f mSec\n", (float)uSec / 1000.0f);


    stbi_write_png(saveTo, width, height, 1, (const void *)data, (width)*1);
    // stbi_write_png(saveTo1, width - 2, height - 2, 1, (const void *)EdgeDetect, (width - 2) * 1);
    stbi_write_png(saveTo1, width - 6, height - 6, 1, (const void *)EdgeDetect, (width - 6) * 1);
    // stbi_write_png(saveTo3, width - 6, height - 6, 1, (const void *)cannyEdge, (width - 6) * 1);
    stbi_write_png(saveTo3, width - 4, height - 4, 1, (const void *)cannyEdge, (width - 4) * 1);
    // stbi_write_png(saveTo2, width - 2, height - 2, 1, (const void *)sobely, (width - 2) * 1);

    // -----------------------------------------------------------
    stbi_image_free(data);
    free(EdgeDetect);
    free(cannyEdge);
    // free(sobely);

    return 0;
}

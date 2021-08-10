
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

    const char *filename = "imgs/original/lena.jpg";
    // const char *filename = "imgs/original/snp2.png";

    char const *saveTo = "imgs/results/data.png";
    char const *saveTo1 = "imgs/results/match.png";
    char const *saveTo2 = "imgs/results/template.png";

    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);

    printf("%s:\nwidth: %d px,\theight: %d px,\tchannels: %d\n", filename, width, height, channels);

    // -----------------------------------------------------------
    unsigned char *template = crop(data, width, height, 1, 190, 241, 190, 241);

    logStartLine("testing 'templateMatch()");
    startTimer();

    unsigned char *templateMatching = templateMatch(data, template, width, height, 51, 51);
    int uSec = endTimer();
    logInfo("'templateMatch()' took, %0.3f mSec\n", (float)uSec / 1000.0f);

    stbi_write_png(saveTo, width, height, 1, (const void *)data, (width)*1);
    stbi_write_png(saveTo1, width - 50, height - 50, 1, (const void *)templateMatching, (width - 50) * 1);
    stbi_write_png(saveTo2, 51, 51, 1, (const void *)template, 51 * 1);

    // -----------------------------------------------------------
    stbi_image_free(data);
    free(templateMatching);
    free(template);

    return 0;
}

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdlib.h>
#include <cstring>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_ref> <image_test>"
                  << std::endl;
        return 1;
    }

    std::string image_path_ref = argv[1];
    std::string image_path_test = argv[2];

    // Load image
    int ref_width, ref_height, ref_channels;
    int test_width, test_height, test_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (ref_image == NULL || test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return 1;
    }

    if (ref_width != test_width || ref_height != test_height || ref_channels != test_channels)
    {
        std::cout << "Error: images are not the same size" << std::endl;
        return 1;
    }

    // Free image
    stbi_image_free(ref_image);
    stbi_image_free(test_image);
    return 0;
}
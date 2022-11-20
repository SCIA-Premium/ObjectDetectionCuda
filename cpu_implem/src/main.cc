#include <cstring>
#include <iostream>
#include <stdlib.h>

#include "images.hh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_image(unsigned char *image, int width, int height,
                const char *filename)
{
    stbi_write_png(filename, width, height, 1, image, width);
}

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
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width,
                                         &ref_height, &ref_channels, 0);
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width,
                                          &test_height, &test_channels, 0);

    if (ref_image == NULL || test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return 1;
    }

    if (ref_width != test_width || ref_height != test_height
        || ref_channels != test_channels)
    {
        std::cout << "Error: images are not the same size" << std::endl;
        return 1;
    }

    // Convert image to grayscale
    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    unsigned char *test_gray = grayscale(test_image, test_width, test_height);

    // Save gray image
    save_image(ref_gray, ref_width, ref_height, "ref_gray.png");
    save_image(test_gray, test_width, test_height, "test_gray.png");

    // Apply Gaussian blur with
    int radius = 2;
    float sigma = 1.0;
    unsigned char *ref_gaussian =
        gaussian_filter(ref_gray, ref_width, ref_height, radius, sigma);
    unsigned char *test_gaussian =
        gaussian_filter(test_gray, test_width, test_height, radius, sigma);

    // Save gaussian image
    save_image(ref_gaussian, ref_width, ref_height, "ref_gaussian.png");
    save_image(test_gaussian, test_width, test_height, "test_gaussian.png");

    // Difference between reference and test image
    unsigned char *diff = difference(ref_gaussian, test_gaussian, ref_width,
                                     ref_height);
    // Save difference image
    save_image(diff, ref_width, ref_height, "diff.png");

    // Morphological opening/closing
    bool closing = true;
    unsigned char *morph_close =
        morphological(diff, ref_width, ref_height, radius, closing);
    unsigned char *morph_open =
        morphological(diff, ref_width, ref_height, radius, !closing);
    unsigned char *morph = morphological(morph_open, ref_width, ref_height,
                                         radius, closing);

    // Save morphological image
    save_image(morph_close, ref_width, ref_height, "morph_close.png");
    save_image(morph_open, ref_width, ref_height, "morph_open.png");
    save_image(morph, ref_width, ref_height, "morph.png");
     

    // Threshold
    unsigned char *thresh = threshold(morph, ref_width, ref_height, 0);

    // Save threshold image
    save_image(thresh, ref_width, ref_height, "thresh.png");
    

    
    // Free images
    stbi_image_free(ref_image);
    stbi_image_free(test_image);
    delete[] ref_gray;
    delete[] test_gray;
    delete[] ref_gaussian;
    delete[] test_gaussian;
    delete[] diff;
    delete[] morph_close;
    delete[] morph_open;
    delete[] morph;
    delete[] thresh;
    return 0;
}
#pragma once
#include <cstddef>
#include <memory>
#include <vector>

struct bounding_box
{
    int x;
    int y;
    int width;
    int height;
};

void grayscale_render(unsigned char *rgbBuffer, unsigned char *grayBuffer, int width, int height, int channels);

void gaussian_blur_render(unsigned char *image, unsigned char *blurImage, int width, int height, float *kernel, int kernelSize);

void difference_render(unsigned char *img1, unsigned char *img2, unsigned char *diff, int width, int height);

void morph_render(unsigned char *img, unsigned char *morph, int width, int height, int kernelRadius, bool closing);

void threshold_render(unsigned char *img, unsigned char *thresh, int width, int height, int threshold);

void ccl_render(const unsigned char *img, unsigned char *ccl, int min_box_size, int min_pixel_value, int width, int height);

void ccl_render_cpu(unsigned char *image, unsigned char *components, int width, int height,
                                    int min_pixel_value, int min_box_size);

void find_bboxes(unsigned char *components, int width, int height,
                 std::vector<bounding_box> &boxes);
#pragma once
#include <cstddef>
#include <memory>

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void grayscale_render(unsigned char *rgbBuffer, unsigned char *grayBuffer, int width, int height, int channels);

void gaussian_blur_render(unsigned char *image, unsigned char *blurImage, int width, int height, float *kernel, int kernelSize);

void difference_render(unsigned char *img1, unsigned char *img2, unsigned char *diff, int width, int height);

void morph_render(unsigned char *img, unsigned char *morph, int width, int height, int kernelRadius, bool closing);

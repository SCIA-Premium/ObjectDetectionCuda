#pragma once
#include <math.h>
#include <cstring>

// Return grayscale image from RGB image
unsigned char *grayscale(unsigned char *image, int width, int height);

// Return Gaussian filter from image
unsigned char *gaussian_filter(unsigned char *image, int width, int height, int radius, float sigma);

// Return difference between two images
unsigned char *difference(unsigned char *image1, unsigned char *image2, int width, int height);
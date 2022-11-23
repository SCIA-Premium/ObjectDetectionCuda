#pragma once
#include <cstring>
#include <math.h>
#include <iostream>

// Return grayscale image from RGB image
unsigned char *grayscale(unsigned char *image, int width, int height);

// Return Gaussian filter from image
unsigned char *gaussian_filter(unsigned char *image, int width, int height,
                               int radius, float sigma);

// Return difference between two images
unsigned char *difference(unsigned char *image1, unsigned char *image2,
                          int width, int height);

// Return morphological closing/opening from image
unsigned char *morphological(unsigned char *image, int width, int height,
                             int radius, bool closing);

// Return thresholded image from image
unsigned char *threshold(unsigned char *image, int width, int height, int threshold);

// Return connected components from image
unsigned char *connected_components(unsigned char *image, int width, int height, int &num_components);

// Draw bounding boxes around components
unsigned char* draw_bbox(unsigned char *image, int width, int height, int min_x, int min_y, int max_x, int max_y);
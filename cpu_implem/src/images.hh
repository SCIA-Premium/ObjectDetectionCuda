#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <vector>

struct bounding_box
{
    int x;
    int y;
    int width;
    int height;
};

// Return grayscale image from RGB image
unsigned char *grayscale(unsigned char *image, int width, int height);

// Return Gaussian filter from image
unsigned char *gaussian_filter(unsigned char *image, int width, int height,
                               int radius, float sigma);

// Return difference between two images
unsigned char *difference(unsigned char *image1, unsigned char *image2,
                          int width, int height);

// Return morphological closing/opening from image
unsigned char *morphological_closing_opening(unsigned char *image, int width,
                                             int height, int opening_radius,
                                             int closing_radius);

// Return thresholded image from image
unsigned char *threshold(unsigned char *image, int width, int height);

// Return connected components from image
unsigned char *connected_components(unsigned char *image, int width, int height,
                                    int min_pixel_value, int min_box_size,
                                    int &num_components);

// Find all bounding boxes of connected components
void find_bboxes(unsigned char *components, int width, int height,
                 std::vector<bounding_box> &boxes, int num_components);

// Draw bounding boxes around components
unsigned char *draw_bbox(unsigned char *image, int width, int height,
                         std::vector<bounding_box> boxes);
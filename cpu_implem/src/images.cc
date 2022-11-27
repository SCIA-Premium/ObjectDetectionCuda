#include "images.hh"

// Return grayscale image from RGB image
unsigned char *grayscale(unsigned char *image, int width, int height)
{
    unsigned char *gray = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++)
    {
        unsigned char r = image[3 * i];
        unsigned char g = image[3 * i + 1];
        unsigned char b = image[3 * i + 2];
        gray[i] = static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    return gray;
}

// Return Gaussian filter from image
unsigned char *gaussian_filter(unsigned char *image, int width, int height,
                               int radius, float sigma)
{
    unsigned char *gaussian = new unsigned char[width * height];
    float *kernel = new float[2 * radius + 1];
    float sum = 0;

    // Populate each element of the kernel with the Gaussian function
    for (int i = -radius; i <= radius; i++)
    {
        kernel[i + radius] =
            exp(-(i * i) / (2 * sigma * sigma)) / (2 * M_PI * (sigma * sigma));
        sum += kernel[i + radius];
    }

    // Normalize the kernel
    for (int i = -radius; i <= radius; i++)
    {
        kernel[i + radius] /= sum;
    }

    memset(gaussian, 0, width * height);

    // Convolve the image with the kernel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                for (int ky = -radius; ky <= radius; ky++)
                {
                    if (i + kx >= 0 && i + kx < height && j + ky >= 0
                        && j + ky < width)
                    {
                        gaussian[i * width + j] +=
                            image[(i + kx) * width + (j + ky)]
                            * kernel[kx + radius] * kernel[ky + radius];
                    }
                }
            }
        }
    }
    delete[] kernel;
    return gaussian;
}

// Return difference between two images
unsigned char *difference(unsigned char *image1, unsigned char *image2, int width, int height)
{
    unsigned char *diff = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++)
    {
        diff[i] = abs(image1[i] - image2[i]);
    }
    return diff;
}

// Return morphological closing/opening from image
unsigned char *morphological(unsigned char *image, int width, int height, int radius, bool closing)
{
    unsigned char *morph = new unsigned char[width * height];
    memset(morph, 0, width * height);

    // Convolve the image with the kernel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                for (int ky = -radius; ky <= radius; ky++)
                {
                    if (i + kx >= 0 && i + kx < height && j + ky >= 0
                        && j + ky < width)
                    {
                        if (closing)
                        {
                            if (image[(i + kx) * width + (j + ky)] > morph[i * width + j])
                            {
                                morph[i * width + j] = image[(i + kx) * width + (j + ky)];
                            }
                            else
                            {
                                morph[i * width + j] = image[i * width + j];
                            }
                        }
                        else
                        {
                            if (image[(i + kx) * width + (j + ky)] < morph[i * width + j])
                            {
                                morph[i * width + j] = image[(i + kx) * width + (j + ky)];
                            }
                            else
                            {
                                morph[i * width + j] = image[i * width + j];
                            }
                        }
                    }
                }
            }
        }
    }
    return morph;
}

// Return thresholded image from image with otsu thresholding and binary thresholding
unsigned char *threshold(unsigned char *image, int width, int height, int threshold)
{
    unsigned char *thresh = new unsigned char[width * height];
    int hist[256];
    memset(hist, 0, 256 * sizeof(int));

    // Calculate histogram
    for (int i = 0; i < width * height; i++)
    {
        hist[image[i]]++;
    }

    // Calculate threshold
    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * hist[i];
    }
    int total = width * height;
    float sumB = 0;
    int q1 = 0;
    int q2 = 0;
    float var_max = 0;
    for (int i = 0; i < 256; i++)
    {
        q1 += hist[i];
        if (q1 == 0)
        {
            continue;
        }
        q2 = total - q1;
        if (q2 == 0)
        {
            break;
        }
        sumB += i * hist[i];
        float m1 = sumB / q1;
        float m2 = (sum - sumB) / q2;
        float var= q1 * q2 * (m1 - m2) * (m1 - m2);
        if (var > var_max)
        {
            threshold = i;
            var_max = var;
        }
    }
    // Threshold the image
    for (int i = 0; i < width * height; i++)
    {
        if (image[i] > threshold)
        {
            thresh[i] = 255;
        }
        else
        {
            thresh[i] = 0;
        }
    }
    return thresh;
}

// Return connected components from image
unsigned char *connected_components(unsigned char *image, int width, int height, int &num_components)
{
    unsigned char *components = new unsigned char[width * height];
    memset(components, 0, width * height);
    int label = 1;
    int *labels = new int[width * height];
    memset(labels, 0, width * height * sizeof(int));
    int *label_map = new int[width * height];
    memset(label_map, 0, width * height * sizeof(int));

    // Label the image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (image[i * width + j] == 255)
            {
                int left = 0;
                int top = 0;
                if (j > 0)
                {
                    left = labels[i * width + j - 1];
                }
                if (i > 0)
                {
                    top = labels[(i - 1) * width + j];
                }
                if (left == 0 && top == 0)
                {
                    labels[i * width + j] = label;
                    label_map[label] = label;
                    label++;
                }
                else if (left != 0 && top == 0)
                {
                    labels[i * width + j] = left;
                }
                else if (left == 0 && top != 0)
                {
                    labels[i * width + j] = top;
                }
                else
                {
                    labels[i * width + j] = std::min(left, top);
                    label_map[std::max(left, top)] = std::min(left, top);
                }
            }
        }
    }

    // Map labels
    for (int i = 0; i < width * height; i++)
    {
        if (labels[i] != 0)
        {
            labels[i] = label_map[labels[i]];
        }
    }

    // Count number of components
    int *count = new int[width * height];
    memset(count, 0, width * height * sizeof(int));
    for (int i = 0; i < width * height; i++)
    {
        if (labels[i] != 0)
        {
            count[labels[i]]++;
        }
    }
    for (int i = 0; i < width * height; i++)
    {
        if (count[i] > 0)
        {
            num_components++;
        }
    }

    // Return components
    for (int i = 0; i < width * height; i++)
    {
        if (labels[i] != 0)
        {
            components[i] = labels[i];
        }
    }
    delete[] labels;
    delete[] label_map;
    delete[] count;
    return components;
}

// Draw bounding boxes around components for rgb image
unsigned char* draw_bbox(unsigned char *image, int width, int height, std::vector<bounding_box> boxes)
{
    unsigned char *bbox = new unsigned char[width * height * 3];
    memcpy(bbox, image, width * height * 3);
    for (auto box : boxes)
    {
        int min_x = box.x;
        int max_x = box.x + box.width;
        int min_y = box.y;
        int max_y = box.y + box.height;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (((i == min_x || i == max_y) && j >= min_x && j <= max_x)
                    || ((j == min_x || j == max_x) && i >= min_y && i <= max_y))
                {
                    bbox[(i * width + j) * 3] = 0;
                    bbox[(i * width + j) * 3 + 1] = 255;
                    bbox[(i * width + j) * 3 + 2] = 0;
                }
            }
        }
    }
   return bbox;
}
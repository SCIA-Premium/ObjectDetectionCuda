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
        gray[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
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
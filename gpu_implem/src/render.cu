#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
{
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

// GPU kernel to convert a rgb image to grayscale
__global__ void grayscale_kernel(const unsigned char *rgb, unsigned char *gray, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int idx_offset = idx * channels;
    unsigned char r = rgb[idx_offset];
    unsigned char g = rgb[idx_offset + 1];
    unsigned char b = rgb[idx_offset + 2];
    gray[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
}

// GPU kernel to add gaussian blur to an image
__global__ void gaussian_blur_kernel(unsigned char *image, unsigned char *blurImage, int width, int height, float *kernel, int kernelRadius)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++)
        {
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++)
            {
                if (y + ky >= 0 && y + ky < height && x + kx >= 0 && x + kx < width)
                {
                    blurImage[y * width + x] += static_cast<unsigned char>(image[(y + ky) * width + (x + kx)] * kernel[kx + kernelRadius] * kernel[ky + kernelRadius]);
                }
            }
        }
    }
}

// // GPU kernel to compute difference between two images
// __global__ void diff_kernel(const std::uint8_t *img1, const std::uint8_t *img2, std::uint8_t *diff, int width, int height)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height)
//     {
//         int idx = y * width + x;
//         diff[idx] = abs(img1[idx] - img2[idx]);
//     }
// }

// Function to render a grayscale image
void grayscale_render(unsigned char *rgbBuffer, unsigned char *grayBuffer, int width, int height, int channels)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devBuffer;

    rc = cudaMalloc(&devBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Copy image to device
    unsigned char *devImage;
    cudaMalloc(&devImage, width * sizeof(unsigned char) * height * channels);
    rc = cudaMemcpy(devImage, rgbBuffer, width * sizeof(unsigned char) * height * channels, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy image to device");

    // Run the kernel with blocks of size 64 x 64
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        // Apply grayscale filter
        grayscale_kernel<<<dimGrid, dimBlock>>>(devImage, devBuffer, width, height, channels);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(grayBuffer, devBuffer, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory rgbImage");
}

// Function to render a gaussian blur image
void gaussian_blur_render(unsigned char *image, unsigned char *blurImage, int width, int height, float *kernel, int kernelSize)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devBuffer;

    rc = cudaMalloc(&devBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Copy image to device
    unsigned char *devImage;
    cudaMalloc(&devImage, width * sizeof(unsigned char) * height);
    rc = cudaMemcpy(devImage, image, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy image to device");

    // Copy kernel to device
    float *devKernel;
    cudaMalloc(&devKernel, kernelSize * sizeof(float) * kernelSize);
    rc = cudaMemcpy(devKernel, kernel, kernelSize * sizeof(float) * kernelSize, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy kernel to device");

    // Run the kernel with blocks of size 64 x 64
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        // Apply gaussian blur filter
        gaussian_blur_kernel<<<dimGrid, dimBlock>>>(devImage, devBuffer, width, height, devKernel, kernelSize);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(blurImage, devBuffer, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory devImage");

    rc = cudaFree(devKernel);
    if (rc)
        abortError("Unable to free memory devKernel");
}

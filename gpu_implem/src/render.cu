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
// __global__ void gaussian_blur_kernel(const std::uint8_t *src, std::uint8_t *dst, int width, int height)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height)
//     {
//         int idx = y * width + x;
//         int sum = 0;
//         int count = 0;

//         for (int i = -1; i <= 1; i++)
//         {
//             for (int j = -1; j <= 1; j++)
//             {
//                 int x1 = x + i;
//                 int y1 = y + j;

//                 if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
//                 {
//                     sum += src[y1 * width + x1];
//                     count++;
//                 }
//             }
//         }

//         dst[idx] = sum / count;
//     }
// }

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

void grayscale_render(unsigned char *rgbBuffer, unsigned char *grayBuffer, int width, int height, int channels)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devBuffer;

    rc = cudaMalloc(&devBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Copy image to device
    unsigned char *rgbImage;
    cudaMalloc(&rgbImage, width * sizeof(unsigned char) * height * channels);
    rc = cudaMemcpy(rgbImage, rgbBuffer, width * sizeof(unsigned char) * height * channels, cudaMemcpyHostToDevice);
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
        grayscale_kernel<<<dimGrid, dimBlock>>>(rgbImage, devBuffer, width, height, channels);

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

    rc = cudaFree(rgbImage);
    if (rc)
        abortError("Unable to free memory rgbImage");
}

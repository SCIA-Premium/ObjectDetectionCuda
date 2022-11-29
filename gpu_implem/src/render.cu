#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <algorithm>

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

// GPU kernel to compute difference between two images
__global__ void diff_kernel(const unsigned char *img1, const unsigned char *img2, unsigned char *diff, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        diff[idx] = abs(img1[idx] - img2[idx]);
    }
}

// Function to render a difference between two images
void difference_render(unsigned char *img1, unsigned char *img2, unsigned char *diff, int width, int height)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devBuffer;

    rc = cudaMalloc(&devBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Copy image to device
    unsigned char *devImage1;
    cudaMalloc(&devImage1, width * sizeof(unsigned char) * height);
    rc = cudaMemcpy(devImage1, img1, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy image to device");

    // Copy image to device
    unsigned char *devImage2;
    cudaMalloc(&devImage2, width * sizeof(unsigned char) * height);
    rc = cudaMemcpy(devImage2, img2, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
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
        // Apply gaussian blur filter
        diff_kernel<<<dimGrid, dimBlock>>>(devImage1, devImage2, devBuffer, width, height);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(diff, devBuffer, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage1);
    if (rc)
        abortError("Unable to free memory devImage1");

    rc = cudaFree(devImage2);
    if (rc)
        abortError("Unable to free memory devImage2");
}

// GPU kernel to compute morphological closing
__global__ void erosion_kernel(const unsigned char *img, unsigned char *morph, int width, int height, int kernelRadius)
{

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;

        unsigned char max = 0;

        for (int i = -kernelRadius; i <= kernelRadius; i++)
        {
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                int x1 = x + i;
                int y1 = y + j;

                if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                {
                    int idx1 = y1 * width + x1;
                    if (img[idx1] > max)
                        max = img[idx1];
                }
            }
        }

        morph[idx] = max;
    }
}

// GPU kernel to compute morphological opening
__global__ void dilation_kernel(const unsigned char *img, unsigned char *morph, int width, int height, int kernelRadius)
{

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;

        unsigned char min = 255;

        for (int i = -kernelRadius; i <= kernelRadius; i++)
        {
            for (int j = -kernelRadius; j <= kernelRadius; j++)
            {
                int x1 = x + i;
                int y1 = y + j;

                if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                {
                    int idx1 = y1 * width + x1;
                    if (img[idx1] < min)
                        min = img[idx1];
                }
            }
        }

        morph[idx] = min;
    }
}

// Function to render a morphological closing/opening
void morph_render(unsigned char *img, unsigned char *morph, int width, int height, int kernelRadius, bool closing)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *firstTransformBuffer;
    unsigned char *secondTransformBuffer;

    rc = cudaMalloc(&firstTransformBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail first transformation buffer allocation");

    rc = cudaMalloc(&secondTransformBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail second transformation buffer allocation");

    // Copy image to device
    unsigned char *devImage;
    cudaMalloc(&devImage, width * sizeof(unsigned char) * height);
    rc = cudaMemcpy(devImage, img, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
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

        // Apply gaussian blur filter
        if (closing)
        {
            // If closing, dilate first then erode
            dilation_kernel<<<dimGrid, dimBlock>>>(devImage, firstTransformBuffer, width, height, kernelRadius);
            erosion_kernel<<<dimGrid, dimBlock>>>(firstTransformBuffer, secondTransformBuffer, width, height, kernelRadius);
        }
        else
        {
            // If opening, erode first then dilate
            erosion_kernel<<<dimGrid, dimBlock>>>(devImage, firstTransformBuffer, width, height, kernelRadius);
            dilation_kernel<<<dimGrid, dimBlock>>>(firstTransformBuffer, secondTransformBuffer, width, height, kernelRadius);
        }

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(morph, secondTransformBuffer, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(firstTransformBuffer);
    if (rc)
        abortError("Unable to free memory firstTransformBuffer");

    rc = cudaFree(secondTransformBuffer);
    if (rc)
        abortError("Unable to free memory secondTransformBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory devImage");
}

// GPU kernel to apply thresholding
__global__ void threshold_kernel(const unsigned char *img, unsigned char *thresh, int width, int height, int threshold)
{

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;

        if (img[idx] < threshold)
            thresh[idx] = 0;
        else
            thresh[idx] = img[idx];
    }
}

// Function to render a thresholded image
void threshold_render(unsigned char *img, unsigned char *thresh, int width, int height, int threshold)
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
    rc = cudaMemcpy(devImage, img, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
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

        // Apply gaussian blur filter
        threshold_kernel<<<dimGrid, dimBlock>>>(devImage, devBuffer, width, height, threshold);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(thresh, devBuffer, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory devImage");
}

/*
// GPU kernel to apply first pass of Connected Component Labeling
__global__ void ccl_kernel1(const unsigned char *img, unsigned char *ccl, std::vector<std::vector<unsigned char>> *equivalency, int width, int height)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = (y + 1) * width + x + 1;

        // Look with 4-connectivity
        if (img[idx] > 0)
        {
            std::vector<unsigned char> neighbors;

            // Check left and top
            if (img[idx - 1] > 0)
                neighbors.push_back(img[idx - 1]);
            if (img[idx - width] > 0)
                neighbors.push_back(ccl[idx - width]);

            // If no neighbors, assign new label
            if (neighbors.size() == 0)
            {
                ccl[idx] = (unsigned char)equivalency->size();
                equivalency->push_back(std::vector<unsigned char>{ccl[idx]});
            }
            else
            {
                // Find smallest label
                unsigned char min = neighbors[0];
                for (size_t i = 1; i < neighbors.size(); i++)
                {
                    if (neighbors[i] < min)
                        min = neighbors[i];
                }

                ccl[idx] = min;

                // Add to equivalency table
                for (auto n : neighbors)
                {
                    if (n != ccl[idx])
                    {
                        // Add to equivalency table
                        (*equivalency)[ccl[idx]].push_back(n);
                        (*equivalency)[n].push_back(ccl[idx]);
                    }
                }
            }
        }
        else
        {
            ccl[idx] = 0;
        }
    }
}

// GPU kernel to apply second pass of Connected Component Labeling
__global__ void ccl_kernel2(unsigned char *ccl, std::vector<std::vector<unsigned char>> *equivalency, int width, int height)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;

        // Look with 4-connectivity
        if (ccl[idx] > 0)
        {
            // Get equivalency list
            std::vector<unsigned char> eq = (*equivalency)[ccl[idx]];

            // Find smallest label
            unsigned char min = eq[0];
            for (size_t i = 1; i < eq.size(); i++)
            {
                if (eq[i] < min)
                    min = eq[i];
            }

            // Update label
            ccl[idx] = min;
        }
    }
}

// Function to render connected component labeling with two passes 8-connectivity
void ccl_render(unsigned char *img, unsigned int *ccl, int width, int height)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devBuffer;

    rc = cudaMalloc(&devBuffer, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Add padding to image
    unsigned char *paddedImage = (unsigned char *)calloc((width + 2) * (height + 2), sizeof(unsigned char));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            paddedImage[(i + 1) * (width + 2) + j + 1] = img[i * width + j];
        }
    }

    // Copy image to device
    unsigned char *devImage;
    cudaMalloc(&devImage, (width + 2) * sizeof(unsigned char) * (height + 2));
    rc = cudaMemcpy(devImage, paddedImage, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy image to device");

    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        // Create equivalency table
        std::vector<std::vector<unsigned char>> equivalency;

        // Apply first pass of Connected Component Labeling
        ccl_kernel1<<<dimGrid, dimBlock>>>(devImage, devBuffer, &equivalency, width, height);

        spdlog::debug("equivalency table size: {}", equivalency.size());

        // Apply second pass of Connected Component Labeling
        ccl_kernel2<<<dimGrid, dimBlock>>>(devBuffer, &equivalency, width, height);

        if (cudaPeekAtLastError())
            abortError("Computation Error");
    }

    // Copy back to main memory
    rc = cudaMemcpy(ccl, devBuffer, width * sizeof(unsigned int) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devBuffer);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory devImage");
}
*/

// Init labels
__global__ void init_labels(unsigned char *img, unsigned char* labels, unsigned char* label_map, int width, int height)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;
        if (img[idx] != 0)
        {
            int left = 0;
            int top = 0;
            if (x > 0)
                left = labels[idx - 1];
            if (y > 0)
                top = labels[idx - width];
            
            if (left == 0 && top == 0)
            {
                labels[idx] = idx + 1;
                label_map[idx] = idx + 1;
            }
            else if (left == 0 && top != 0)
            {
                labels[idx] = top;
            }
            else if (top == 0 && left != 0)
            {
                labels[idx] = left;
            }
            else
            {
                if (left < top)
                {
                    labels[idx] = left;
                    label_map[top] = left;
                }
                else
                {
                    labels[idx] = top;
                    label_map[left] = top;
                }
            }
        }
    }
}

// Update labels
__global__ void update_labels(unsigned char *img, unsigned char* labels, unsigned char* label_map, int *count, unsigned char* max, int width, int height)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;
        if (labels[idx] != 0)
        {
            labels[idx] = label_map[labels[idx]];
            count[labels[idx]]++;
            if (max[labels[idx]] < img[idx])
                max[labels[idx]] = img[idx];
        }
    }
}

// Clean labels
__global__ void clean_labels(unsigned char* labels, unsigned char* label_map, int *count, unsigned char* max, int min_box_size, int min_pixel_value, int width, int height)
{
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;
        if (labels[idx] != 0)
        {
            if (count[labels[idx]] < min_box_size || max[labels[idx]] < min_pixel_value)
                labels[idx] = 0;
        }
    }
}
// Function to render connected component labeling
void ccl_render(const unsigned char *img, unsigned char *ccl, int min_box_size, int min_pixel_value, int width, int height)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    unsigned char *devLabels;

    rc = cudaMalloc(&devLabels, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");
    
    unsigned char *devLabels_map;

    rc = cudaMalloc(&devLabels_map, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");

    // Add padding to image
    /*unsigned char *paddedImage = (unsigned char *)calloc((width + 2) * (height + 2), sizeof(unsigned char));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            paddedImage[(i + 1) * (width + 2) + j + 1] = img[i * width + j];
        }
    }*/

    int *devCount;
    rc = cudaMalloc(&devCount, width * sizeof(int) * height);
    if (rc)
        abortError("Fail buffer allocation");
    cudaMemset(devCount, 0, width * height * sizeof(int));
    unsigned char *devMax;
    rc = cudaMalloc(&devMax, width * sizeof(unsigned char) * height);
    if (rc)
        abortError("Fail buffer allocation");
    cudaMemset(devMax, 0, width * height * sizeof(unsigned char));

    // Copy image to device
    unsigned char *devImage;
    cudaMalloc(&devImage, (width) * sizeof(unsigned char) * (height));
    rc = cudaMemcpy(devImage, img, width * sizeof(unsigned char) * height, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail copy image to device");
    else
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        // Apply Connected Component Labeling
        init_labels<<<dimGrid, dimBlock>>>(devImage, devLabels, devLabels_map, width, height);

        // Count number of different labels
        /*int *labels_count;
        cudaMalloc(&labels_count, (width) * sizeof(int) * (height));
        cudaMemset(labels_count, 0, width * height * sizeof(int));
        spdlog::debug("test");
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                spdlog::debug("test");
                if (devLabels[i * width + j] != 0)
                    labels_count[devLabels[i * width + j]]++;
            }
        }
        int num_labels = 0;
        for (int i = 0; i < width * height; i++)
        {
            if (labels_count[i] != 0)
                num_labels++;
        }
        spdlog::debug("number of labels: {}", num_labels);
        */

        //update_labels<<<dimGrid, dimBlock>>>(devImage, devLabels, devLabels_map, devCount, devMax, width, height);
        
        //clean_labels<<<dimGrid, dimBlock>>>(devLabels, devLabels_map, devCount, devMax, min_box_size, min_pixel_value, width, height);
        if (cudaPeekAtLastError())
            abortError("Computation Error");

    }

    // Copy back to main memory
    rc = cudaMemcpy(ccl, devLabels, width * sizeof(unsigned char) * height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devLabels);
    if (rc)
        abortError("Unable to free memory devBuffer");

    rc = cudaFree(devImage);
    if (rc)
        abortError("Unable to free memory devImage");
    
    rc = cudaFree(devLabels_map);
    if (rc)
        abortError("Unable to free memory devLabels_map");
    
    rc = cudaFree(devCount);
    if (rc)
        abortError("Unable to free memory devCount");

    rc = cudaFree(devMax);
    if (rc)
        abortError("Unable to free memory devMax");

    //free(paddedImage);
}
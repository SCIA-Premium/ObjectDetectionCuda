#include "images.hpp"
#include <iostream>
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

        /*
        // Count number of different labels
        int *labels_count = (int*) calloc((width) * (height) + 1, sizeof(int));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                std::cout << devLabels[i * width + j] << std::endl;
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
}

// Return connected components from image
void ccl_render_cpu(unsigned char *image, unsigned char *components, int width, int height,
                                    int min_pixel_value, int min_box_size)
{
    unsigned char *label_map = new unsigned char[width * height];
    memset(label_map, 0, width * height);

    // Label the image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (image[i * width + j] != 0)
            {
                int left = 0;
                int top = 0;
                if (j > 0)
                {
                    left = components[i * width + j - 1];
                }
                if (i > 0)
                {
                    top = components[(i - 1) * width + j];
                }

                if (left == 0 && top == 0)
                {
                    components[i * width + j] = i * width + j + 1;
                    label_map[i * width + j] = i * width + j + 1;
                }
                else if (left != 0 && top == 0)
                {
                    components[i * width + j] = left;
                }
                else if (left == 0 && top != 0)
                {
                    components[i * width + j] = top;
                }
                else
                {
                    components[i * width + j] = std::min(left, top);
                    label_map[std::max(left, top)] = std::min(left, top);
                }
            }
        }
    }

    // Map labels by updating their values, count the number of occurences and
    // find the max pixel value of each component
    int *count = new int[width * height];
    memset(count, 0, width * height * sizeof(int));
    unsigned char *max = new unsigned char[width * height];
    memset(max, 0, width * height);
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0)
        {
            components[i] = label_map[components[i]];
            count[components[i]]++;
            max[components[i]] = std::max(max[components[i]], image[i]);
        }
    }

    // Remove small components or components with max pixel value less than
    // min_pixel_value
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0
            && (count[components[i]] < min_box_size
                || max[components[i]] < min_pixel_value))
        {
            components[i] = 0;
        }
    }
    // Free memory
    delete[] label_map;
    delete[] count;
    delete[] max;
}

// Find bounding boxes for each components
void find_bboxes(unsigned char *components, int width, int height,
                 std::vector<bounding_box> &boxes)
{
    int *min_x = new int[width * height];
    int *min_y = new int[width * height];
    int *max_x = new int[width * height];
    int *max_y = new int[width * height];
    memset(min_x, 0, width * height * sizeof(int));
    memset(min_y, 0, width * height * sizeof(int));
    memset(max_x, 0, width * height * sizeof(int));
    memset(max_y, 0, width * height * sizeof(int));

    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0)
        {
            int x = i % width;
            int y = i / width;
            if (min_x[components[i]] == 0)
            {
                min_x[components[i]] = x;
            }
            if (min_y[components[i]] == 0)
            {
                min_y[components[i]] = y;
            }
            if (max_x[components[i]] == 0)
            {
                max_x[components[i]] = x;
            }
            if (max_y[components[i]] == 0)
            {
                max_y[components[i]] = y;
            }
            min_x[components[i]] = std::min(min_x[components[i]], x);
            min_y[components[i]] = std::min(min_y[components[i]], y);
            max_x[components[i]] = std::max(max_x[components[i]], x);
            max_y[components[i]] = std::max(max_y[components[i]], y);
        }
    }
    // Create bounding boxes
    unsigned char *components_map = new unsigned char[width * height];
    memset(components_map, 0, width * height);
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0 && components_map[components[i]] == 0)
        {
            bounding_box box;
            box.x = min_x[components[i]];
            box.y = min_y[components[i]];
            box.width = max_x[components[i]] - min_x[components[i]] + 1;
            box.height = max_y[components[i]] - min_y[components[i]] + 1;
            boxes.push_back(box);
            components_map[components[i]] = components[i];
        }
    }
    delete[] min_x;
    delete[] min_y;
    delete[] max_x;
    delete[] max_y;
    delete[] components_map;
}
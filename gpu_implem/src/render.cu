#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

// GPU kernel to convert a rgb image to grayscale
__global__ void grayscale_kernel(const std::uint8_t* rgb, std::uint8_t* gray, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    const std::uint8_t& pixel = rgb[idx];
    gray[idx] = 0.21 * pixel.r + 0.71 * pixel.g + 0.07 * pixel.b;
  }
}

void step1(char* hostBuffer, int width, int height, std::ptrdiff_t stride)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    grayscale_kernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}

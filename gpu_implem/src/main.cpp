#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "render.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_png(const std::byte* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}

// Load all image of folders with stb 
void load_images(const std::string& folder, std::vector<image>& images)
{
  std::filesystem::path path(folder);
  for (const auto& entry : std::filesystem::directory_iterator(path))
  {
    if (entry.is_regular_file())
    {
      int width, height, channels;
      auto* data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 4);
      if (data)
      {
        images.push_back({width, height, channels, data});
      }
    }
  }
}

// Usage: ./main
int main(int argc, char** argv)
{
  std::string output_folder = "/output";
  std::string ref_image = "/ref.png";
  std::string input_folder = "/input";


  CLI::App app{"main"};
  app.add_option("-o", output_folder, "Output Folder");
  app.add_option("-r", ref_image, "Reference Image");
  app.add_option("-i", input_folder, "Input Folder");

  CLI11_PARSE(app, argc, argv);

  // Load input images
  std::vector<image> images;
  load_images(input_folder, images);

  // Load reference image
  int width, height, channels;
  auto* ref_data = stbi_load(ref_image.c_str(), &width, &height, &channels, 4);

  // Create buffer
  constexpr int kRGBASize = 4;
  int stride = width * kRGBASize;
  auto buffer = std::make_unique<std::byte[]>(height * stride);

  step1(reinterpret_cast<char*>(buffer.get()), width, height, stride, niter);

  // Save
  write_png(buffer.get(), width, height, stride, filename.c_str());
  spdlog::info("Output saved in {}.", filename);
}


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

// Load all image of folders with stb
void load_images(const std::string &folder, std::vector<unsigned char *> &images)
{
    int imageCount = 0;
    int imageProcessed = 0;
    std::filesystem::path path(folder);

    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        imageCount++;
        if (entry.is_regular_file())
        {
            spdlog::info("Loaded image {}.", entry.path().string().c_str());
            int width, height, channels;
            unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char *));
            data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 3);
            if (data == NULL)
            {
                spdlog::error("Failed to load image {}", entry.path().string());
                continue;
            }
            imageProcessed++;
            images.push_back(data);
        }
    }

    spdlog::info("Loaded {} images out of {}.", imageProcessed, imageCount);
}

// Save all the images in a folder
void save_images(const std::string &folder, std::vector<unsigned char *> &images, int width, int height, int channels)
{
    std::filesystem::path path(folder);
    std::filesystem::create_directory(path);

    for (size_t i = 0; i < images.size(); i++)
    {
        std::string filename = (i == 0) ? path.string() + "/ref.png" : path.string() + "/input_" + std::to_string(i) + ".png";
        stbi_write_png(filename.c_str(), width, height, channels, images[i], width * channels);
    }
}

// Function to apply the grayscale filter
void grayscale(std::vector<unsigned char *> &inputImages, std::vector<unsigned char *> &outputImages, int width, int height, int channels)
{
    for (unsigned char *image : inputImages)
    {
        unsigned char *grayscaleImage = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (grayscaleImage == NULL)
        {
            spdlog::error("Failed to allocate memory for grayscale reference image");
            continue;
        }

        grayscale_render(image, grayscaleImage, width, height, channels);
        outputImages.push_back(grayscaleImage);
    }
}

// Usage: ./main
int main(int argc, char **argv)
{
    std::string output_folder = "/output";
    std::string ref_image = "/ref.png";
    std::string input_folder = "/input";

    CLI::App app{"main"};
    app.add_option("-o", output_folder, "Output Folder");
    app.add_option("-r", ref_image, "Reference Image");
    app.add_option("-i", input_folder, "Input Folder");

    CLI11_PARSE(app, argc, argv);

    // Load reference image
    int width, height, channels;
    unsigned char *ref_data = stbi_load(ref_image.c_str(), &width, &height, &channels, 3);
    if (ref_data == NULL)
    {
        spdlog::error("Failed to load image {}", ref_image);
        return 1;
    }

    // Store images in a vector
    std::vector<unsigned char *> images;
    images.push_back(ref_data);

    // Load input images
    load_images(input_folder, images);

    // Render grayscale on all images
    std::vector<unsigned char *> grayscaleImages;
    grayscale(images, grayscaleImages, width, height, channels);

    // Save reference image
    save_images(output_folder, grayscaleImages, width, height, 1);
    spdlog::info("Output saved in {}.", output_folder);

    // Save all images
    return 0;
}
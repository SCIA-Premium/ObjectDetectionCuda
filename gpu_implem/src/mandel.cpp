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
    int image_count = 0;
    int image_processed = 0;
    std::filesystem::path path(folder);

    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        image_count++;
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
            image_processed++;
            images.push_back(data);
        }
    }

    spdlog::info("Loaded {} images out of {}.", image_processed, image_count);
}

// Save all the images in a folder
void save_images(const std::string &folder, std::vector<unsigned char *> &images, int width, int height, int channels, std::string &prefix)
{
    std::filesystem::path path(folder);
    std::filesystem::create_directory(path);

    for (size_t i = 0; i < images.size(); i++)
    {
        std::string filename = path.string() + "/";
        if (i == 0 && prefix != "difference_" && prefix != "morph_closing_" && prefix != "morph_opening_")
        {
            filename += prefix + "ref.png";
        }
        else
        {
            filename += prefix + "input_" + std::to_string(i) + ".png";
        }

        stbi_write_png(filename.c_str(), width, height, channels, images[i], width * channels);
    }
}

// Function to apply the grayscale filter
void grayscale(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &output_images, int width, int height, int channels)
{
    for (unsigned char *image : input_images)
    {
        unsigned char *grayscale_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));
        if (grayscale_image == NULL)
        {
            spdlog::error("Failed to allocate memory for grayscale reference image");
            continue;
        }

        grayscale_render(image, grayscale_image, width, height, channels);
        output_images.push_back(grayscale_image);
    }
}

// Function to apply gaussian blur filter
void gaussian_blur(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &output_images, int width, int height)
{
    // Compute the kernel for the gaussian blur
    int radius = 4;
    float sigma = 1.5;
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

    // Apply the gaussian blur filter
    for (unsigned char *image : input_images)
    {
        unsigned char *gaussian_blur_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (gaussian_blur_image == NULL)
        {
            spdlog::error("Failed to allocate memory for gaussian blur reference image");
            continue;
        }

        gaussian_blur_render(image, gaussian_blur_image, width, height, kernel, radius);
        output_images.push_back(gaussian_blur_image);
    }
}

// Function to apply difference between reference and input images
void difference(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &output_images, int width, int height)
{
    for (size_t i = 1; i < input_images.size(); i++)
    {
        unsigned char *difference_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));
        if (difference_image == NULL)
        {
            spdlog::error("Failed to allocate memory for difference image");
            continue;
        }

        difference_render(input_images[0], input_images[i], difference_image, width, height);
        output_images.push_back(difference_image);
    }
}

// Function to apply morphological erosion filter
void morphological_erosion(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &closing_images, std::vector<unsigned char *> &output_images, int width, int height)
{
    int closing_radius = 10;
    int opening_radius = 20;

    for (unsigned char *image : input_images)
    {
        unsigned char *morphological_closing_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));
        if (morphological_closing_image == NULL)
        {
            spdlog::error("Failed to allocate memory for morphological erosion reference image");
            continue;
        }

        morph_render(image, morphological_closing_image, width, height, closing_radius, true);
        closing_images.push_back(morphological_closing_image);

        unsigned char *morphological_opening_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));
        morph_render(morphological_closing_image, morphological_opening_image, width, height, opening_radius, false);
        output_images.push_back(morphological_opening_image);
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
    std::vector<unsigned char *> grayscale_images;
    grayscale(images, grayscale_images, width, height, channels);

    // Save grayscale images
    std::string prefix = "grayscale_";
    save_images(output_folder, grayscale_images, width, height, 1, prefix);

    // Apply the gaussian blur on all images
    std::vector<unsigned char *> gaussian_blur_images;
    gaussian_blur(grayscale_images, gaussian_blur_images, width, height);

    // Save gaussian blur image
    prefix = "gaussian_blur_";
    save_images(output_folder, gaussian_blur_images, width, height, 1, prefix);

    // Apply the difference between the reference image and the input images
    std::vector<unsigned char *> difference_images;
    difference(gaussian_blur_images, difference_images, width, height);

    // Save difference images
    prefix = "difference_";
    save_images(output_folder, difference_images, width, height, 1, prefix);

    // Apply the morphological erosion on all images
    std::vector<unsigned char *> morphological_closing_images;
    std::vector<unsigned char *> morphological_opening_images;
    morphological_erosion(difference_images, morphological_closing_images, morphological_opening_images, width, height);

    // Save morphological closing images
    prefix = "morph_closing_";
    save_images(output_folder, morphological_closing_images, width, height, 1, prefix);

    // Save morphological opening images
    prefix = "morph_opening_";
    save_images(output_folder, morphological_opening_images, width, height, 1, prefix);

    spdlog::info("Output saved in {}.", output_folder);

    // Save all images
    return 0;
}

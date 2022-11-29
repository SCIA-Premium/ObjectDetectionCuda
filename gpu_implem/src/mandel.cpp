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

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Load all image of folders with stb
void load_images(const std::string &folder, std::vector<unsigned char *> &images, std::vector<std::string> &images_paths)
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
            images_paths.push_back(entry.path().string());
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
        if (i == 0 && prefix != "difference_" && prefix != "morph_closing_" && prefix != "morph_opening_" && prefix != "threshold_")
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
        unsigned char *grayscale_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
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
        unsigned char *difference_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
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
        unsigned char *morphological_closing_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (morphological_closing_image == NULL)
        {
            spdlog::error("Failed to allocate memory for morphological erosion reference image");
            continue;
        }

        morph_render(image, morphological_closing_image, width, height, closing_radius, true);
        closing_images.push_back(morphological_closing_image);

        unsigned char *morphological_opening_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        morph_render(morphological_closing_image, morphological_opening_image, width, height, opening_radius, false);
        output_images.push_back(morphological_opening_image);
    }
}

// Function to compute the histogram of each image
void histogram(std::vector<unsigned char *> &input_images, std::vector<unsigned int *> &histograms, int width, int height)
{
    for (unsigned char *image : input_images)
    {
        unsigned int *histogram_image = (unsigned int *)calloc(256, sizeof(unsigned int));
        if (histogram_image == NULL)
        {
            spdlog::error("Failed to allocate memory for histogram reference image");
            continue;
        }

        // Loop through the image and compute the histogram
        for (int i = 0; i < width * height; i++)
        {
            histogram_image[image[i]]++;
        }

        histograms.push_back(histogram_image);
    }
}

// Function to compute the threshold of each image with Otsu method and histogram
void compute_threshold(unsigned int *hist, int *threshold, int width, int height)
{
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
        float var = q1 * q2 * (m1 - m2) * (m1 - m2);
        if (var > var_max)
        {
            *threshold = i;
            var_max = var;
        }
    }
}

// Function to apply the threshold filter
void threshold(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &output_images, int width, int height)
{
    // Compute the histogram of each image
    std::vector<unsigned int *> histograms;
    histogram(input_images, histograms, width, height);

    // Compute the threshold of each image and apply the threshold filter
    for (size_t i = 0; i < input_images.size(); i++)
    {
        unsigned char *threshold_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (threshold_image == NULL)
        {
            spdlog::error("Failed to allocate memory for threshold reference image");
            continue;
        }

        int threshold = 80;
        compute_threshold(histograms[i], &threshold, width, height);

        threshold_render(input_images[i], threshold_image, width, height, threshold);
        output_images.push_back(threshold_image);
    }
}

// Function to apply connected components filter
void connected_components(std::vector<unsigned char *> &input_images, std::vector<unsigned char *> &output_images, int min_box_size, int min_pixel_value, int width, int height)
{
    for (unsigned char *image : input_images)
    {
        unsigned char *connected_components_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (connected_components_image == NULL)
        {
            spdlog::error("Failed to allocate memory for connected components reference image");
            continue;
        }

        ccl_render(image, connected_components_image, min_box_size, min_pixel_value, width, height);
        output_images.push_back(connected_components_image);
    }
}

struct bounding_box
{
    int x;
    int y;
    int width;
    int height;
};

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

// Compute the bounding boxes of each image
void bounding_boxes(std::vector<unsigned char *> &input_images, std::vector<std::string> &images_paths, int width, int height, json *j)
{
    for (size_t i = 1; i < input_images.size(); i++)
    {
        std::vector<bounding_box> boxes;
        find_bboxes(input_images[i - 1], width, height, boxes);
        auto boxes_array = std::vector<std::vector<int>>();
        for (bounding_box box : boxes)
        {
            auto res = std::vector<int>{box.x, box.y, box.width, box.height};
            boxes_array.push_back(res);
        }
        (*j)[images_paths[i]] = boxes_array;
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
    std::vector<std::string> images_paths;
    images.push_back(ref_data);
    images_paths.push_back(ref_image);

    // Load input images
    load_images(input_folder, images, images_paths);

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

    // Compute the threshold of each image
    std::vector<unsigned char *> threshold_images;
    threshold(morphological_opening_images, threshold_images, width, height);

    // Save threshold images
    prefix = "threshold_";
    save_images(output_folder, threshold_images, width, height, 1, prefix);

    // Compute the connected components of each image
    int min_box_size = 30;
    int min_pixel_value = 30;
    std::vector<unsigned char *> connected_components_images;
    connected_components(threshold_images, connected_components_images, min_box_size, min_pixel_value, width, height);
    /*std::vector<unsigned char *> connected_components_images;
    connected_components(threshold_images, connected_components_images, width, height);

*/
    //Save connected components images
    prefix = "connected_components_";
    save_images(output_folder, connected_components_images, width, height, 1, prefix);
    spdlog::info("Output saved in {}.", output_folder);

    // Compute the bounding boxes of each image
    json j;
    bounding_boxes(connected_components_images, images_paths, width, height, &j);

    std::cout << j.dump(4) << std::endl;
    // Save all images
    return 0;
}

#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdlib.h>
#include <unistd.h>

#include "images.hh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using json = nlohmann::json;

static void save_image(unsigned char *image, int width, int height,
                const char *filename)
{
    stbi_write_png(filename, width, height, 1, image, width);
}

static void save_rgb_image(unsigned char *image, int width, int height,
                    const char *filename)
{
    stbi_write_png(filename, width, height, 3, image, width * 3);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0]
                  << " --save <image_ref> <image_test> [image_test...]" << std::endl;
        return 1;
    }

    bool save = false;
    if (!strcmp(argv[1], "--save"))
    {
        save = true;
        argv++;
        argc--;
    }
    json j;

    // Parameters
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;
    int opening_radius = 10;
    int closing_radius = 10;
    int num_components = 0;
    int min_pixel_value = 30;
    int min_box_size = 30;

    // Load ref image
    std::string image_path_ref = argv[1];
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width,
                                         &ref_height, &ref_channels, 0);

    for (int i = 2; i < argc; i++)
    {
        // Load test image
        std::string image_path_test = argv[i];
        int test_width, test_height, test_channels;
        unsigned char *test_image =
            stbi_load(image_path_test.c_str(), &test_width, &test_height,
                      &test_channels, 0);

        if (ref_image == NULL || test_image == NULL)
        {
            std::cout << "Error: cannot load image" << std::endl;
            return 1;
        }

        if (ref_width != test_width || ref_height != test_height
            || ref_channels != test_channels)
        {
            std::cout << "Error: images are not the same size" << std::endl;
            return 1;
        }
        // Convert image to grayscale
        unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
        unsigned char *test_gray =
            grayscale(test_image, test_width, test_height);

        // Apply Gaussian blur withi
        unsigned char *ref_gaussian = gaussian_filter(
            ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);
        unsigned char *test_gaussian =
            gaussian_filter(test_gray, test_width, test_height, gaussian_radius,
                            gaussian_sigma);

        // Difference between reference and test image
        unsigned char *diff =
            difference(ref_gaussian, test_gaussian, ref_width, ref_height);

        // Morphological closing/opening
        unsigned char *morph = morphological_closing_opening(
            diff, ref_width, ref_height, opening_radius, closing_radius);

        // Threshold
        unsigned char *thresh =
            threshold(morph, ref_width, ref_height);

        // Connected components
        unsigned char *components =
            connected_components(thresh, ref_width, ref_height, min_pixel_value,
                                 min_box_size, num_components);

        // Find all bounding boxes of connected components
        std::vector<bounding_box> boxes;
        find_bboxes(components, ref_width, ref_height, boxes, num_components);

        // Save images
        if (save)
        {
            // Draw all bounding box
            unsigned char *bbox =
                draw_bbox(test_image, ref_width, ref_height, boxes);
            // Save grayscaled images
            save_image(ref_gray, ref_width, ref_height, "ref_gray.png");
            save_image(test_gray, test_width, test_height, "test_gray.png");
            // Save gaussian images
            save_image(ref_gaussian, ref_width, ref_height, "ref_gaussian.png");
            save_image(test_gaussian, test_width, test_height,
                       "test_gaussian.png");
            // Save difference image
            save_image(diff, ref_width, ref_height, "diff.png");
            // Save morphological image
            save_image(morph, ref_width, ref_height, "morph.png");
            // Save thresholded image
            save_image(thresh, ref_width, ref_height, "thresh.png");
            // Save connected components image
            save_image(components, ref_width, ref_height, "components.png");
            // Save bounding box image
            save_rgb_image(bbox, ref_width, ref_height, "bbox.png");
            delete[] bbox;
        }

        // Output json
        auto boxes_json = std::vector<std::vector<int>>();
        for (auto box : boxes)
        {
            auto res =
                std::vector<int>({ box.x, box.y, box.width, box.height });
            boxes_json.push_back(res);
        }
        j[image_path_test] = boxes_json;

        // Free images
        stbi_image_free(test_image);
        delete[] ref_gray;
        delete[] test_gray;
        delete[] ref_gaussian;
        delete[] test_gaussian;
        delete[] diff;
        delete[] morph;
        delete[] thresh;
        delete[] components;
    }
    stbi_image_free(ref_image);
    std::cout << j.dump(4) << std::endl;
    return 0;
}
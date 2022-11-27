#include <cstring>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include "images.hh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using json = nlohmann::json;

void save_image(unsigned char *image, int width, int height,
                const char *filename)
{
    stbi_write_png(filename, width, height, 1, image, width);
}

void save_rgb_image(unsigned char *image, int width, int height,
                    const char *filename)
{
    stbi_write_png(filename, width, height, 3, image, width * 3);
}



// Find bounding boxes with highest peaks for components
void find_bboxes(unsigned char *components, int width, int height, std::vector<bounding_box> &bboxes)
{
    int *label_map = new int[width * height];
    memset(label_map, 0, width * height * sizeof(int));
    int label = 1;
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0)
        {
            if (label_map[components[i]] == 0)
            {
                label_map[components[i]] = label;
                label++;
            }
            components[i] = label_map[components[i]];
        }
    }

    // Count number of components
    int *count = new int[width * height];
    memset(count, 0, width * height * sizeof(int));
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0)
        {
            count[components[i]]++;
        }
    }
    for (int i = 0; i < width * height; i++)
    {
        if (count[i] > 0)
        {
            bounding_box box;
            box.x = width;
            box.y = height;
            box.width = 0;
            box.height = 0;
            bboxes.push_back(box);
        }
    }

    // Find bounding boxes
    for (int i = 0; i < width * height; i++)
    {
        if (components[i] != 0)
        {
            int x = i % width;
            int y = i / width;
            bounding_box box = bboxes[components[i] - 1];
            if (x < box.x)
            {
                box.x = x;
            }
            if (x > box.x + box.width)
            {
                box.width = x - box.x;
            }
            if (y < box.y)
            {
                box.y = y;
            }
            if (y > box.y + box.height)
            {
                box.height = y - box.y;
            }
            bboxes[components[i] - 1] = box;
        }
    }
    // Select bounding boxes with highest peaks
    std::vector<bounding_box> selected_bboxes;
    for (auto box : bboxes)
    {
        int max_peak = 0;
        int max_peak_x = 0;
        int max_peak_y = 0;
        for (int i = box.y; i < box.y + box.height; i++)
        {
            for (int j = box.x; j < box.x + box.width; j++)
            {
                if (components[i * width + j] != 0)
                {
                    if (count[components[i * width + j]] > max_peak)
                    {
                        max_peak = count[components[i * width + j]];
                        max_peak_x = j;
                        max_peak_y = i;
                    }
                }
            }
        }
        if (max_peak_x > 0 && max_peak_y > 0)
        {
            bounding_box selected_box;
            selected_box.x = max_peak_x;
            selected_box.y = max_peak_y;
            selected_box.width = 0;
            selected_box.height = 0;
            selected_bboxes.push_back(selected_box);
        }

    }
    bboxes = selected_bboxes;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        
        std::cout << "Usage: " << argv[0] << " <image_ref> <image_test> [image_test...]"
                  << std::endl;
        return 1;
    }
    bool save = false;
    json j;

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


        // Apply Gaussian blur with
        int radius = 2;
        float sigma = 1.0;
        unsigned char *ref_gaussian =
            gaussian_filter(ref_gray, ref_width, ref_height, radius, sigma);
        unsigned char *test_gaussian =
            gaussian_filter(test_gray, test_width, test_height, radius, sigma);

        // Difference between reference and test image
        unsigned char *diff =
            difference(ref_gaussian, test_gaussian, ref_width, ref_height);

        // Morphological opening/closing
        bool closing = true;
        unsigned char *morph_close =
            morphological(diff, ref_width, ref_height, radius, closing);
        unsigned char *morph_open =
            morphological(diff, ref_width, ref_height, radius, !closing);
        unsigned char *morph =
            morphological(morph_open, ref_width, ref_height, radius, closing);

        // Threshold
        unsigned char *thresh = threshold(morph, ref_width, ref_height, 0);

        // Connected components
        int num_components = 0;
        unsigned char *components =
            connected_components(thresh, ref_width, ref_height, num_components);

        // Find all bounding boxes of connected components
        std::vector<bounding_box> boxes;
        find_bboxes(components, ref_width, ref_height, boxes);

        /*
        // Compute the bounding box around all components
        int min_x = ref_width;
        int min_y = ref_height;
        int max_x = 0;
        int max_y = 0;
        for (int i = 0; i < ref_width * ref_height; i++)
        {
            if (components[i] != 0)
            {
                int x = i % ref_width;
                int y = i / ref_width;
                if (x < min_x)
                    min_x = x;
                if (x > max_x)
                    max_x = x;
                if (y < min_y)
                    min_y = y;
                if (y > max_y)
                    max_y = y;
            }
        }
        */
        // std::cout << "Bounding box: (" << min_x << ", " << min_y << ") - ("
        //          << max_x << ", " << max_y << ")" << std::endl;
        // Draw bounding box
        unsigned char *bbox = draw_bbox(test_image, ref_width, ref_height,
                                       boxes);

        // Save images
        if (save)
        {
            // Save gray image
            save_image(ref_gray, ref_width, ref_height,  "ref_gray.png");
            save_image(test_gray, test_width, test_height, "test_gray.png");
            // Save gaussian image
            save_image(ref_gaussian, ref_width, ref_height, "ref_gaussian.png");
            save_image(test_gaussian, test_width, test_height,
                       "test_gaussian.png");
            // Save difference image
            save_image(diff, ref_width, ref_height, "diff.png");
            // Save morphological image
            save_image(morph_close, ref_width, ref_height, "morph_close.png");
            save_image(morph_open, ref_width, ref_height, "morph_open.png");
            save_image(morph, ref_width, ref_height, "morph.png");
            // Save thresholded image
            save_image(thresh, ref_width, ref_height, "thresh.png");
            // Save connected components image
            save_image(components, ref_width, ref_height, "components.png");
            // Save bounding box image
            save_rgb_image(bbox, ref_width, ref_height, "bbox.png");
        }

        // Output json
        auto boxes_json = std::vector<std::vector<int>>();
        for (auto box: boxes)
        {
            auto res = std::vector<int>({box.x, box.y, box.width, box.height});
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
        delete[] morph_close;
        delete[] morph_open;
        delete[] morph;
        delete[] thresh;
        delete[] components;
        delete[] bbox;
    }
    stbi_image_free(ref_image);
    std::cout << j.dump(4) << std::endl;
    return 0;
}
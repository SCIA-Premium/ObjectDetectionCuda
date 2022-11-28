#include "images.hh"

// Return grayscale image from RGB image
unsigned char *grayscale(unsigned char *image, int width, int height)
{
    unsigned char *gray = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++)
    {
        unsigned char r = image[3 * i];
        unsigned char g = image[3 * i + 1];
        unsigned char b = image[3 * i + 2];
        gray[i] =
            static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    return gray;
}

// Return Gaussian filter from image
unsigned char *gaussian_filter(unsigned char *image, int width, int height,
                               int radius, float sigma)
{
    unsigned char *gaussian = new unsigned char[width * height];
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

    memset(gaussian, 0, width * height);

    // Convolve the image with the kernel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                for (int ky = -radius; ky <= radius; ky++)
                {
                    if (i + kx >= 0 && i + kx < height && j + ky >= 0
                        && j + ky < width)
                    {
                        gaussian[i * width + j] +=
                            image[(i + kx) * width + (j + ky)]
                            * kernel[kx + radius] * kernel[ky + radius];
                    }
                }
            }
        }
    }
    delete[] kernel;
    return gaussian;
}

// Return difference between two images
unsigned char *difference(unsigned char *image1, unsigned char *image2,
                          int width, int height)
{
    unsigned char *diff = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++)
    {
        diff[i] = abs(image1[i] - image2[i]);
    }
    return diff;
}

// Function to apply morphological erosion
static void morphological_erosion(unsigned char *image, unsigned char *res,
                                  int width, int height, int radius)
{
    // Convolve the image with the kernel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            unsigned char max = 0;
            for (int kx = -radius; kx <= radius; kx++)
            {
                for (int ky = -radius; ky <= radius; ky++)
                {
                    int x1 = j + kx;
                    int y1 = i + ky;

                    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                    {
                        max = std::max(max, image[y1 * width + x1]);
                    }
                }
            }
            res[i * width + j] = max;
        }
    }
}

// Function to apply morphological dilation
static void morphological_dilation(unsigned char *image, unsigned char *res,
                                   int width, int height, int radius)
{
    // Convolve the image with the kernel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            unsigned char min = 255;

            for (int kx = -radius; kx <= radius; kx++)
            {
                for (int ky = -radius; ky <= radius; ky++)
                {
                    int x1 = j + kx;
                    int y1 = i + ky;

                    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                    {
                        min = std::min(min, image[y1 * width + x1]);
                    }
                }
            }
            res[i * width + j] = min;
        }
    }
}

// Function to apply morphological closing
static void morphological_closing(unsigned char *image, int width, int height,
                                  int radius)
{
    unsigned char *res = new unsigned char[width * height];
    morphological_dilation(image, res, width, height, radius);
    morphological_erosion(res, image, width, height, radius);
    delete[] res;
}

// Function to apply morphological opening
static void morphological_opening(unsigned char *image, int width, int height,
                                  int radius)
{
    unsigned char *res = new unsigned char[width * height];
    morphological_erosion(image, res, width, height, radius);
    morphological_dilation(res, image, width, height, radius);
    delete[] res;
}

// Function to return morphological opening and closing from image
unsigned char *morphological_opening_closing(unsigned char *image, int width,
                                             int height, int opening_radius,
                                             int closing_radius)
{
    unsigned char *morph = new unsigned char[width * height];
    memcpy(morph, image, width * height);
    morphological_closing(morph, width, height, closing_radius);
    morphological_opening(morph, width, height, opening_radius);
    return morph;
}

// Return thresholded image from image with threshold value
unsigned char *threshold(unsigned char *image, int width, int height,
                         int threshold)
{
    unsigned char *thresh = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++)
    {
        if (image[i] > threshold)
        {
            thresh[i] = image[i];
        }
        else
        {
            thresh[i] = 0;
        }
    }
    return thresh;
}

// Return connected components from image
unsigned char *connected_components(unsigned char *image, int width, int height,
                                    int min_pixel_value, int min_box_size,
                                    int &num_components)
{
    unsigned char *components = new unsigned char[width * height];
    memset(components, 0, width * height);
    unsigned char label = 1;
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
                    components[i * width + j] = label;
                    label_map[label] = label;
                    label++;
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
    int *count = new int[label];
    memset(count, 0, label * sizeof(int));
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

    num_components = label;
    return components;
}

// Find bounding boxes from components
void find_bboxes(unsigned char *components, int width, int height,
                 std::vector<bounding_box> &boxes, int num_components)
{
    // Find bounding boxes for each components
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
    unsigned char *components_map = new unsigned char[num_components];
    memset(components_map, 0, num_components);
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

// Draw bounding boxes around components for rgb image
unsigned char *draw_bbox(unsigned char *image, int width, int height,
                         std::vector<bounding_box> boxes)
{
    unsigned char *bbox = new unsigned char[width * height * 3];
    memcpy(bbox, image, width * height * 3);
    for (auto box : boxes)
    {
        int min_x = box.x;
        int max_x = box.x + box.width;
        int min_y = box.y;
        int max_y = box.y + box.height;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (((i == min_x || i == max_y) && j >= min_x && j <= max_x)
                    || ((j == min_x || j == max_x) && i >= min_y && i <= max_y))
                {
                    bbox[(i * width + j) * 3] = 0;
                    bbox[(i * width + j) * 3 + 1] = 255;
                    bbox[(i * width + j) * 3 + 2] = 0;
                }
            }
        }
    }
    return bbox;
}
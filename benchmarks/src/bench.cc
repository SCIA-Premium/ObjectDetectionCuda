#include <benchmark/benchmark.h>

#include "images.hh"
#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <opencv2/opencv.hpp>
 
std::string image_path_ref = "../../samples/SCIA_Premium/0.jpg";

std::string image_path_test = "../../samples/SCIA_Premium/120.jpg";

// Benchmark for the reference grayscale function
static void BM_Grayscale(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    for (auto _ : state)
        grayscale(ref_image, ref_width, ref_height);
}

// Benchmark for the opencv grayscale function
static void BM_Grayscale_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }
    
    for (auto _ : state)
    {
        cv::Mat gray_ref;
        cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    }
}

// Benchmark for the gaussian_filter function
static void BM_Gaussian_Filter(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;

    for (auto _ : state)
        gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);
}

// Benchmark for the opencv gaussian blur function
static void BM_Gaussian_Filter_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat gray_ref;
    cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    int gaussian_radius = 5;
    float gaussian_sigma = 0;
    
    for (auto _ : state)
    {
        cv::Mat gaussian_ref;
        cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);
    }
}

// Benchmark for the difference function
static void BM_Difference(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    int test_width, test_height, test_channels;
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;

    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    unsigned char *ref_gaussian = gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);

    unsigned char *test_gray = grayscale(test_image, test_width, test_height);
    unsigned char *test_gaussian = gaussian_filter(test_gray, test_width, test_height, gaussian_radius, gaussian_sigma);

    for (auto _ : state)
        difference(ref_gaussian, test_gaussian, ref_width, ref_height);
}

// Benchmark for the opencv difference function
static void BM_Difference_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat test_image = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (test_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    int gaussian_radius = 5;
    float gaussian_sigma = 0;

    cv::Mat gray_ref;
    cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_ref;
    cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat gray_test;
    cv::cvtColor(test_image, gray_test, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_test;
    cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);
    
    for (auto _ : state)
    {
        cv::Mat diff;
        cv::absdiff(gaussian_ref, gaussian_test, diff);
    }
}

// Benchmark for the Opening Closing function
static void BM_Opening_Closing(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    int test_width, test_height, test_channels;
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;
    int opening_radius = 10;
    int closing_radius = 10;

    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    unsigned char *ref_gaussian = gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);

    unsigned char *test_gray = grayscale(test_image, test_width, test_height);
    unsigned char *test_gaussian = gaussian_filter(test_gray, test_width, test_height, gaussian_radius, gaussian_sigma);

    unsigned char *diff = difference(ref_gaussian, test_gaussian, ref_width, ref_height);

    for (auto _ : state)
        morphological_closing_opening(diff, ref_width, ref_height, opening_radius, closing_radius);
}

// Benchmark for the opencv Opening Closing function
static void BM_Opening_Closing_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat test_image = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (test_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    int gaussian_radius = 5;
    float gaussian_sigma = 0;
    int opening_radius = 20;
    int closing_radius = 20;

    cv::Mat gray_ref;
    cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_ref;
    cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat gray_test;
    cv::cvtColor(test_image, gray_test, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_test;
    cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat diff;
    cv::absdiff(gaussian_ref, gaussian_test, diff);
    
    for (auto _ : state)
    {
        cv::Mat morphological;
        cv::Mat opening_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_radius, opening_radius));
        cv::Mat closing_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing_radius, closing_radius));
        cv::morphologyEx(diff, morphological, cv::MORPH_CLOSE, closing_radius);
        cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, opening_radius);
    }
}

// Benchmark for the Threshold function
static void BM_Threshold(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    int test_width, test_height, test_channels;
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;
    int opening_radius = 10;
    int closing_radius = 10;
    int threshold_value = 10;

    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    unsigned char *ref_gaussian = gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);

    unsigned char *test_gray = grayscale(test_image, test_width, test_height);
    unsigned char *test_gaussian = gaussian_filter(test_gray, test_width, test_height, gaussian_radius, gaussian_sigma);

    unsigned char *diff = difference(ref_gaussian, test_gaussian, ref_width, ref_height);
    unsigned char *morph = morphological_closing_opening(diff, ref_width, ref_height, opening_radius, closing_radius);

    for (auto _ : state)
        threshold(morph, ref_width, ref_height, threshold_value);
}

// Benchmark for the opencv Threshold function
static void BM_Threshold_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat test_image = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (test_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    int gaussian_radius = 5;
    float gaussian_sigma = 0;
    int opening_radius = 20;
    int closing_radius = 20;
    int threshold = 80;

    cv::Mat gray_ref;
    cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_ref;
    cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat gray_test;
    cv::cvtColor(test_image, gray_test, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_test;
    cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat diff;
    cv::absdiff(gaussian_ref, gaussian_test, diff);

    cv::Mat morphological;
    cv::Mat opening_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_radius, opening_radius));
    cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing_radius, closing_radius));
    cv::morphologyEx(diff, morphological, cv::MORPH_CLOSE, closing_radius);
    cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, opening_radius);
    
    for (auto _ : state)
    {
        cv::Mat thresh;
        cv::threshold(morphological, thresh, threshold, 255, cv::THRESH_TOZERO);
    }
}

// Benchmark for the Connected Component function
static void BM_Connected_Component(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    int test_width, test_height, test_channels;
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }
    
    int gaussian_radius = 2;
    float gaussian_sigma = 1.0;
    int opening_radius = 10;
    int closing_radius = 10;
    int threshold_value = 10;
    int num_components = 0;
    int min_pixel_value = 30;
    int min_box_size = 30;

    unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
    unsigned char *ref_gaussian = gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);

    unsigned char *test_gray = grayscale(test_image, test_width, test_height);
    unsigned char *test_gaussian = gaussian_filter(test_gray, test_width, test_height, gaussian_radius, gaussian_sigma);

    unsigned char *diff = difference(ref_gaussian, test_gaussian, ref_width, ref_height);
    unsigned char *morph = morphological_closing_opening(diff, ref_width, ref_height, opening_radius, closing_radius);
    unsigned char *thresh = threshold(morph, ref_width, ref_height, threshold_value);

    for (auto _ : state)
        connected_components(thresh, ref_width, ref_height, min_pixel_value, min_box_size, num_components);
}

// Benchmark for the opencv Connected Component function
static void BM_Connected_Component_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat test_image = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (test_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    int gaussian_radius = 5;
    float gaussian_sigma = 0;
    int opening_radius = 20;
    int closing_radius = 20;
    int threshold = 80;

    cv::Mat gray_ref;
    cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_ref;
    cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat gray_test;
    cv::cvtColor(test_image, gray_test, cv::COLOR_BGR2GRAY);
    cv::Mat gaussian_test;
    cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

    cv::Mat diff;
    cv::absdiff(gaussian_ref, gaussian_test, diff);

    cv::Mat morphological;
    cv::Mat opening_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_radius, opening_radius));
    cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing_radius, closing_radius));
    cv::morphologyEx(diff, morphological, cv::MORPH_CLOSE, closing_radius);
    cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, opening_radius);

    cv::Mat thresh;
    cv::threshold(morphological, thresh, threshold, 255, cv::THRESH_TOZERO);
    
    for (auto _ : state)
    {
        cv::Mat labels, stats, centroids;
        cv::Mat components = thresh.clone();
        cv::connectedComponentsWithStats(components, labels, stats, centroids);
    }
}

// Benchmark for the complete pipeline
static void BM_Pipeline(benchmark::State& state) {
    int ref_width, ref_height, ref_channels;
    unsigned char *ref_image = stbi_load(image_path_ref.c_str(), &ref_width, &ref_height, &ref_channels, 0);

    if (ref_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    int test_width, test_height, test_channels;
    unsigned char *test_image = stbi_load(image_path_test.c_str(), &test_width, &test_height, &test_channels, 0);

    if (test_image == NULL)
    {
        std::cout << "Error: cannot load image" << std::endl;
        return;
    }

    for (auto _ : state)
    {
        int gaussian_radius = 2;
        float gaussian_sigma = 1.0;
        int opening_radius = 10;
        int closing_radius = 10;
        int threshold_value = 10;
        int num_components = 0;
        int min_pixel_value = 30;
        int min_box_size = 30;

        unsigned char *ref_gray = grayscale(ref_image, ref_width, ref_height);
        unsigned char *ref_gaussian = gaussian_filter(ref_gray, ref_width, ref_height, gaussian_radius, gaussian_sigma);

        unsigned char *test_gray = grayscale(test_image, test_width, test_height);
        unsigned char *test_gaussian = gaussian_filter(test_gray, test_width, test_height, gaussian_radius, gaussian_sigma);

        unsigned char *diff = difference(ref_gaussian, test_gaussian, ref_width, ref_height);
        unsigned char *morph = morphological_closing_opening(diff, ref_width, ref_height, opening_radius, closing_radius);
        unsigned char *thresh = threshold(morph, ref_width, ref_height, threshold_value);

        connected_components(thresh, ref_width, ref_height, min_pixel_value, min_box_size, num_components);
    }
}

// Benchmark for the opencv pipeline
static void BM_Pipeline_CV(benchmark::State& state) {
    cv::Mat ref_image = cv::imread(image_path_ref, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (ref_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    cv::Mat test_image = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (test_image.empty())
    {
        std::cout << "Could not load image" << std::endl;
        return;
    }

    for (auto _ : state)
    {
        int gaussian_radius = 5;
        float gaussian_sigma = 0;
        int opening_radius = 20;
        int closing_radius = 20;
        int threshold = 80;

        cv::Mat gray_ref;
        cv::cvtColor(ref_image, gray_ref, cv::COLOR_BGR2GRAY);
        cv::Mat gaussian_ref;
        cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

        cv::Mat gray_test;
        cv::cvtColor(test_image, gray_test, cv::COLOR_BGR2GRAY);
        cv::Mat gaussian_test;
        cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

        cv::Mat diff;
        cv::absdiff(gaussian_ref, gaussian_test, diff);

        cv::Mat morphological;
        cv::Mat opening_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_radius, opening_radius));
        cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing_radius, closing_radius));
        cv::morphologyEx(diff, morphological, cv::MORPH_CLOSE, closing_radius);
        cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, opening_radius);

        cv::Mat thresh;
        cv::threshold(morphological, thresh, threshold, 255, cv::THRESH_TOZERO);

        cv::Mat labels, stats, centroids;
        cv::Mat components = thresh.clone();
        cv::connectedComponentsWithStats(components, labels, stats, centroids);
    }
}

BENCHMARK(BM_Grayscale);
BENCHMARK(BM_Grayscale_CV);

BENCHMARK(BM_Gaussian_Filter);
BENCHMARK(BM_Gaussian_Filter_CV);

BENCHMARK(BM_Difference);
BENCHMARK(BM_Difference_CV);

BENCHMARK(BM_Opening_Closing);
BENCHMARK(BM_Opening_Closing_CV);

BENCHMARK(BM_Threshold);
BENCHMARK(BM_Threshold_CV);

BENCHMARK(BM_Connected_Component);
BENCHMARK(BM_Connected_Component_CV);

BENCHMARK(BM_Pipeline);
BENCHMARK(BM_Pipeline_CV);

BENCHMARK_MAIN();
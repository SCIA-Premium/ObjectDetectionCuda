#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

int main(int argc, char const *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0]
                  << "--show <image_ref> <image_test> [image_test...]"
                  << std::endl;
        return 1;
    }

    bool show = false;
    if (!strcmp(argv[1], "--show"))
    {
        show = true;
        argv++;
        argc--;
    }
    json j;

    // Load ref image
    std::string image_path_ref = argv[1];
    cv::Mat image_ref = cv::imread(image_path_ref, cv::IMREAD_COLOR);
    if (image_ref.empty())
    {
        std::cout << "Could not read the image: " << image_path_ref << std::endl;
        return 1;
    }

    // Parameters
    int gaussian_radius = 5;
    float gaussian_sigma = 0;
    int opening_radius = 20;
    int closing_radius = 20;
    int threshold = 80;
    int min_pixel_value = 30;
    int min_box_size = 30;

    for (int i = 2; i < argc; i++)
    {
        // Load test image
        std::string image_path_test = argv[i];
        cv::Mat image_test = cv::imread(image_path_test, cv::IMREAD_COLOR);

        // Check if image is loaded
        if (image_test.empty())
        {
            std::cout << "Could not load image" << std::endl;
            return 1;
        }
        // Convert image to grayscale
        cv::Mat gray_ref, gray_test;
        cv::cvtColor(image_ref, gray_ref, cv::COLOR_BGR2GRAY);
        cv::cvtColor(image_test, gray_test, cv::COLOR_BGR2GRAY);

        // Apply Gaussian filter on image
        cv::Mat gaussian_ref, gaussian_test;
        cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);
        cv::GaussianBlur(gray_test, gaussian_test, cv::Size(gaussian_radius, gaussian_radius), gaussian_sigma);

        // Compute the difference between two images
        cv::Mat difference;
        cv::absdiff(gaussian_ref, gaussian_test, difference);

        // Perform morphological closing/opening with a disk to noise
        cv::Mat morphological;
        cv::Mat opening_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening_radius, opening_radius));
        cv::Mat closing_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closing_radius, closing_radius));
        cv::morphologyEx(difference, morphological, cv::MORPH_CLOSE, closing_radius);
        cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, opening_radius);

        // Compute the threshold of the image
        cv::Mat thresh;
        cv::threshold(morphological, thresh, threshold, 255,
                    cv::THRESH_TOZERO);

        // Get connected components
        cv::Mat labels, stats, centroids;
        cv::Mat components = thresh.clone();
        cv::connectedComponentsWithStats(components, labels, stats, centroids);

        // Remove from components connected components with number of pixels < min_box_size
        for (int i = 0; i < stats.rows; i++)
        {
            int *params = stats.ptr<int>(i);
            if (params[cv::ConnectedComponentsTypes::CC_STAT_AREA] < min_box_size)
            {
                components.setTo(0, labels == i);
            }
        }

        // Remove from components connected components with max pixel value < min_pixel_value
        for (int i = 0; i < stats.rows; i++)
        {
            cv::Mat mask = labels == i;
            cv::Mat component = components & mask;
            double min, max;
            cv::minMaxLoc(component, &min, &max);
            if (max < min_pixel_value)
            {
                components.setTo(0, mask);
            }
        }

        // Recompute connected components

        cv::connectedComponentsWithStats(components, labels, stats, centroids);

        // Compute the bounding box of the blobs
        cv::Mat boundingBox = image_test.clone();
        auto boxes_json = std::vector<std::vector<int>>();
        for (int i = 1; i < stats.rows; i++)
        {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            boxes_json.push_back({x, y, width, height});
            if (show)
                cv::rectangle(boundingBox, cv::Point(x, y),
                        cv::Point(x + width, y + height), cv::Scalar(0, 255, 0),
                        2);
        }
        j[image_path_test] = boxes_json;
        // Display images
        if (show)
        {
            cv::imshow("Image reference", image_ref);
            cv::imshow("Image test", image_test);
            cv::imshow("Gaussian ref", gaussian_ref);
            cv::imshow("Gaussian test", gaussian_test);
            cv::imshow("Difference", difference);
            cv::imshow("Morphological", morphological);
            cv::imshow("Threshold", thresh);
            cv::imshow("Components", components);
            cv::imshow("Bounding box", boundingBox);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

    }
    std::cout << j.dump(4) << std::endl;
    return 0;
}

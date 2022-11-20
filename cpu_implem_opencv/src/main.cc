
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_ref> <image_test>" << std::endl;
        return 1;
    }

    std::string image_path_ref = argv[1];
    std::string image_path_test = argv[2];

    // Load image
    cv::Mat image_ref = cv::imread(image_path_ref, cv::IMREAD_COLOR);
    cv::Mat image_test = cv::imread(image_path_test, cv::IMREAD_COLOR);

    // Check if image is loaded
    if (image_ref.empty() || image_test.empty())
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
    cv::GaussianBlur(gray_ref, gaussian_ref, cv::Size(5, 5), 0);
    cv::GaussianBlur(gray_test, gaussian_test, cv::Size(5, 5), 0);

    // Compute the difference between two images
    cv::Mat difference;
    cv::absdiff(gaussian_ref, gaussian_test, difference);

    // Perform morphological closing/opening with a disk to noise
    cv::Mat morphological;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(difference, morphological, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(morphological, morphological, cv::MORPH_OPEN, kernel);

    // Compute the threshold of the image
    cv::Mat threshold;
    cv::threshold(morphological, threshold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Keep only the blobs with high peaks
    cv::Mat peak;
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(threshold, labels, stats, centroids);

    // Compute the bounding box of the blobs
    cv::Mat boundingBox = image_test.clone();
    for (int i = 1; i < stats.rows; i++)
    {  
        std::cout << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        cv::rectangle(boundingBox, cv::Point(x, y), cv::Point(x + width, y + height), cv::Scalar(0, 255, 0), 2);
    }

    // Display images
    cv::imshow("Image reference", image_ref);
    cv::imshow("Image test", image_test);
    cv::imshow("Difference", difference);
    cv::imshow("Threshold", threshold);
    cv::imshow("Bounding box", boundingBox);
    cv::waitKey(0);

    return 0;
}

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>


std::vector<cv::Point2f> convertToPointVector(const float* points, int n) {
    std::vector<cv::Point2f> pointVec;
    pointVec.reserve(n);
    for (int i = 0; i < n; ++i) {
        pointVec.emplace_back(points[2 * i], points[2 * i + 1]);
    }
    return pointVec;
}


void DLT(cv::Mat &img1, cv::Mat &img2) {
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error Loading Images!" << std::endl;
        return;
    }

    // get image dimentions
    int im1_width = img1.cols;
    int im1_height = img1.rows;
    std::cout << "Image 1 dim: (" << im1_height << ", " << im1_width << ")" << std::endl;

    int im2_width = img2.cols;
    int im2_height = img2.rows;
    std::cout << "Image 2 dim: (" << im2_height << ", " << im2_width << ")" << std::endl;
    // define points relative to their dimensions
    //int n = sizeof(points) / sizeof(float);
    std::vector<cv::Point2f> points1 = 
    {
        cv::Point2f(0, 0),              // top left corner
        cv::Point2f(img1.cols - 1, 0),   // top right corner
        cv::Point2f(0, img1.rows - 1),
        cv::Point2f(img1.cols - 1, img1.rows - 1)
    };
    std::vector<cv::Point2f> points2 = 
    {
        cv::Point2f(img2.cols * 0.0f,  img2.rows * 0.33f),
        cv::Point2f(img2.cols * 0.85f, img2.rows * 0.25f),
        cv::Point2f(img2.cols * 0.15f, img2.rows * 0.7f),
        cv::Point2f(100.0000009536743, 62)
    };
    // ensure enough correspondence points (n = 4)
    if (points1.size() < 4 || points2.size() < 4) {
        std::cerr << "Needs at least 4 points for DLT correspondence." << std::endl;
        return;
    }
    // calculate homography with RANSAC
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);
    // check H validity
    if (H.empty()) {
        std::cerr << "Homography calculation failed." << std::endl;
        return;
    }
    // Display the homography matrix
    std::cout << "Homography Matrix (H):\n" << H << std::endl;

    // Apply homography transformation to img1 to align with img2
    cv::Mat transformed_img;
    cv::warpPerspective(img1, transformed_img, H, img2.size());

    // Display images
    cv::imshow("Image 1", img1);
    cv::imshow("Transformed 1", transformed_img);
    cv::imshow("Image 2", img2);
    cv::waitKey(0);
}


int main(int argc, char const *argv[])
{
    cv::Mat im1 = cv::imread("calibration_images/apple-light.jpg");
    cv::Mat im2 = cv::imread("calibration_images/apple-light-transformed.jpg");
    DLT(im1, im2);
    return 0;
}

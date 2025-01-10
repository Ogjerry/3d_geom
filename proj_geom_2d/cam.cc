/*
 * Affine Transformation. Keeping track of correspondence points
 * during affine transformation. 
*/



#include <iostream>
#include <opencv4/opencv2/opencv.hpp>



int main(int argc, char const *argv[])
{
    // Load the image
    cv::Mat src = cv::imread("calibration_images/apple-light.jpg");
    if (src.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Define the source points and destination points for the transformation
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    // Define points from the source image
    srcTri[0] = cv::Point2f(0, 0);              // top left corner
    srcTri[1] = cv::Point2f(src.cols - 1, 0);   // top right corner
    srcTri[2] = cv::Point2f(0, src.rows - 1);   // bottom left corner

    // Define points for the transformed image
    dstTri[0] = cv::Point2f(src.cols * 0.0f, src.rows * 0.33f);
    dstTri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.7f);

    // Get the affine transform matrix
    cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);

    cv::Point2f srcFourthPoint(src.cols - 1, src.rows - 1);
    // transforming point2f to a (3,1) matrix cv::Mat
    cv::Mat srcFourthMat = (cv::Mat_<double>(3,1) << srcFourthPoint.x, srcFourthPoint.y, 1.0);
    cv::Mat bottomRight = warpMat * srcFourthMat;
    std::cout << "Transformed Bottom Right Corner: " << bottomRight << std::endl;

    // Create an output image
    cv::Mat warpDst;
    cv::warpAffine(src, warpDst, warpMat, src.size());

    // save transformed image
    cv::imwrite("calibration_images/apple-light-transformed.jpg", warpDst);

    // Display the images
    cv::imshow("Source Image", src);
    cv::imshow("Warped Image", warpDst);

    // Wait for a key press and exit
    cv::waitKey(0);
    return 0;
}
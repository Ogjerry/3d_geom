#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

// Function to compute Plücker coordinates for a line from two 2D points in homogeneous coordinates
cv::Vec3d computePlucker2D(const cv::Vec3d& P1, const cv::Vec3d& P2) {
    // Plücker coordinates for 2D projective space: l12, l13, l23
    cv::Vec3d plucker;
    
    plucker[0] = P1[0] * P2[1] - P1[1] * P2[0];  // l12
    plucker[1] = P1[0] * P2[2] - P1[2] * P2[0];  // l13
    plucker[2] = P1[1] * P2[2] - P1[2] * P2[1];  // l23
    
    return plucker;
}

int main() {
    // Image (apple.jpg) would be loaded here if needed, but for now we'll use hardcoded points
    // Example: Randomly selected 2D points from the image
    cv::Point2d p1(100, 50);   // Point 1 in pixel coordinates (x1, y1)
    cv::Point2d p2(200, 150);  // Point 2 in pixel coordinates (x2, y2)

    // Convert 2D image points to homogeneous coordinates (adding 1 as the third coordinate)
    cv::Vec3d P1_hom = cv::Vec3d(p1.x, p1.y, 1.0);  // P1 = (x1, y1, w=1)
    cv::Vec3d P2_hom = cv::Vec3d(p2.x, p2.y, 1.0);  // P2 = (x2, y2, w=1)

    // Compute Plücker coordinates for the line passing through P1 and P2 in 2D
    cv::Vec3d plucker2D = computePlucker2D(P1_hom, P2_hom);

    // Display the computed Plücker coordinates
    std::cout << "Plücker coordinates for the line (2D):" << std::endl;
    std::cout << "l12: " << plucker2D[0] << std::endl;
    std::cout << "l13: " << plucker2D[1] << std::endl;
    std::cout << "l23: " << plucker2D[2] << std::endl;

    return 0;
}

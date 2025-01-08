#include "readimage.h"
#include <iostream>

bool readImageAsVector(const std::string& imagePath, std::vector<float>& outputVector) {
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return false;
    }

    // Resize the image to 32x32
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(32, 32));

    // Convert the resized image into a vector and normalize the pixel values
    outputVector.clear();
    for (int row = 0; row < resizedImage.rows; ++row) {
        for (int col = 0; col < resizedImage.cols; ++col) {
            // Get the BGR pixel values
            cv::Vec3b pixel = resizedImage.at<cv::Vec3b>(row, col);
            
            // Normalize each channel to the range [0, 1]
            outputVector.push_back(static_cast<float>(pixel[0]) / 255.0f); // Blue channel
            outputVector.push_back(static_cast<float>(pixel[1]) / 255.0f); // Green channel
            outputVector.push_back(static_cast<float>(pixel[2]) / 255.0f); // Red channel
        }
    }

    return true;
}

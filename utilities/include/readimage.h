#ifndef READIMAGE_H
#define READIMAGE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Reads an image from a file, resizes it to 32x32, and converts it to a vector of floats.
 * @param imagePath The path to the image file.
 * @param outputVector The vector to store the resulting 32x32x3 float values.
 * @return True if successful, false if the image could not be loaded.
 */
bool readImageAsVector(const std::string& imagePath, std::vector<float>& outputVector);

#endif // READIMAGE_H

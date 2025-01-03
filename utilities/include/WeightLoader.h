#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>

class WeightLoader {
public:
    // Loads a binary file into a vector of floats
    static std::vector<float> loadWeights(const std::string& filePath);

    // Print the weights (for debugging)
    static void printWeights(const std::vector<float>& weights, size_t numPerRow = 10);
};

#endif // WEIGHT_LOADER_H

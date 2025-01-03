#include "WeightLoader.h"

std::vector<float> WeightLoader::loadWeights(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // Determine the file size
    std::streamsize size = file.tellg();
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of float size: " + filePath);
    }

    file.seekg(0, std::ios::beg);

    // Create a buffer to hold the weights
    std::vector<float> weights(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(weights.data()), size)) {
        throw std::runtime_error("Failed to read the file: " + filePath);
    }

    file.close();
    return weights;
}

void WeightLoader::printWeights(const std::vector<float>& weights, size_t numPerRow) {
    size_t count = 0;
    for (float weight : weights) {
        std::cout << weight << " ";
        if (++count % numPerRow == 0) {
            std::cout << std::endl;
        }
    }
    if (weights.size() % numPerRow != 0) {
        std::cout << std::endl; // Add a new line if the last row wasn't full
    }
}

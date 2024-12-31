#include "flatten.h"
#include <iostream>

std::vector<float> Flatten::flatten_2d(const std::vector<std::vector<float>>& input) {
    std::vector<float> output;
    for (const auto& row : input) {
        output.insert(output.end(), row.begin(), row.end());
    }
    return output;
}

std::vector<float> Flatten::forward(const std::vector<std::vector<float>>& input) {
    // Call flatten function for 2D input
    return flatten_2d(input);
}
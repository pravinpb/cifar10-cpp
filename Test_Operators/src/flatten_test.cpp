#include "flatten.h"
#include <iostream>

int main() {
    Flatten flattenLayer;

    // Example 2D input (3x3 matrix)
    std::vector<std::vector<float>> input = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    // Flatten the input
    std::vector<float> output = flattenLayer.forward(input);

    // Print the flattened output
    for (float value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

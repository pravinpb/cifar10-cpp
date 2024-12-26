#include "conv2d.h"
#include <iostream>
#include <vector>

int main() {
    Conv2D conv(1, 1, 3);  // Example: 1 input channel, 1 output channel, 3x3 kernel.

    std::vector<std::vector<std::vector<float>>> input = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    std::vector<std::vector<std::vector<std::vector<float>>>> weights = {{{{1, 0, -1}, {1, 0, -1}, {1, 0, -1}}}};
    std::vector<float> biases = {0};

    auto output = conv.forward(input, weights, biases);

    for (const auto& row : output[0]) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

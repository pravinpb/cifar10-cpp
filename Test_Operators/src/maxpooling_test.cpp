#include "maxpooling.h"
#include <iostream>

int main() {
    MaxPooling pool(2, 2);  // 2x2 pooling window, 2x2 stride

    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    std::vector<std::vector<float>> output = pool.apply_pooling(input);

    for (const auto& row : output) {
        for (float value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

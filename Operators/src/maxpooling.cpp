
#include "maxpooling.h"
#include <algorithm>
#include <iostream>

MaxPooling::MaxPooling(int pool_size, int stride) 
    : pool_size(pool_size), stride(stride) {}

int MaxPooling::calculate_output_dim(int input_dim) {
    return (input_dim - pool_size) / stride + 1;
}

std::vector<std::vector<float>> MaxPooling::apply_pooling(const std::vector<std::vector<float>>& input) {
    int input_height = input.size();
    int input_width = input[0].size();

    int output_height = calculate_output_dim(input_height);
    int output_width = calculate_output_dim(input_width);

    std::vector<std::vector<float>> output(output_height, std::vector<float>(output_width, 0));

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float max_val = input[i * stride][j * stride];  // Start with the first element in the pool
            // Loop over the pool window
            for (int m = 0; m < pool_size; ++m) {
                for (int n = 0; n < pool_size; ++n) {
                    int row = i * stride + m;
                    int col = j * stride + n;
                    if (row < input_height && col < input_width) {
                        max_val = std::max(max_val, input[row][col]);
                    }
                }
            }
            output[i][j] = max_val;
        }
    }

    return output;
}

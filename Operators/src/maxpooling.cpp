#include "maxpooling.h"
#include <algorithm>
#include <iostream>

MaxPooling::MaxPooling(int pool_size, int stride)  // Default pool size to 2
    : pool_size(pool_size), stride(stride) {}

int MaxPooling::calculate_output_dim(int input_dim) {
    return (input_dim - pool_size) / stride + 1;
}


std::vector<std::vector<std::vector<float>>> MaxPooling::apply_pooling(const std::vector<std::vector<std::vector<float>>>& input) {
    int input_height = input[0].size();  // Height of the input
    int input_width = input[0][0].size();  // Width of the input
    int num_channels = input.size();  // Number of channels

    int output_height = calculate_output_dim(input_height);
    int output_width = calculate_output_dim(input_width);

    // Create output with the same number of channels as input initially
    std::vector<std::vector<std::vector<float>>> output(num_channels, 
        std::vector<std::vector<float>>(output_height, 
            std::vector<float>(output_width, 0)));

    // Apply max pooling for each channel independently
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                float max_val = input[c][i * stride][j * stride];  // Start with the first element in the pool
                for (int m = 0; m < pool_size; ++m) {
                    for (int n = 0; n < pool_size; ++n) {
                        int row = i * stride + m;
                        int col = j * stride + n;
                        if (row < input_height && col < input_width) {
                            max_val = std::max(max_val, input[c][row][col]);
                        }
                    }
                }
                output[c][i][j] = max_val;  // Store the max value in the corresponding channel
            }
        }
    }

    return output;  // Return pooled output with 15x15x96 shape
}

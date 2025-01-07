// dense.cpp
#include "dense.h"
#include "relu.h"
#include "softmax.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>

void dense(const std::vector<float>& input, const std::vector<float>& weights,
           const std::vector<float>& bias, std::vector<float>& output,
           const std::array<int, 2>& input_shape, const std::array<int, 2>& output_shape,
           const std::string& activation, const std::string layer_name) {
    int input_size = input_shape[1];
    int output_size = output_shape[1];

    assert(input.size() == input_size && "Input size mismatch for dense layer.");
    assert(weights.size() == input_size * output_size && "Weights size mismatch for dense layer.");
    assert(bias.size() == output_size && "Bias size mismatch for dense layer.");

    std::cout << "Performing dense operation for layer: " << layer_name << std::endl;
    for (int o = 0; o < output_size; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * output_size + o];
        }
        output[o] = sum + bias[o];
    }

    if (activation == "relu") {
        relu(output);
    } else if (activation == "softmax") {
        softmax(output);
    }

}

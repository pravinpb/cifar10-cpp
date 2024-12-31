#include "dense.h"
#include <iostream>
#include <cmath>

Dense::Dense(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
    // Initialize weights and biases as empty vectors
    weights.resize(output_size, std::vector<float>(input_size, 0.0f));
    biases.resize(output_size, 0.0f);
}

void Dense::set_weights(const std::vector<std::vector<float>>& new_weights) {
    weights = new_weights;
}

void Dense::set_biases(const std::vector<float>& new_biases) {
    biases = new_biases;
}

std::vector<float> Dense::forward(const std::vector<float>& input) {
    std::vector<float> output(output_size, 1.0f);

    // Perform matrix multiplication: output = weights * input + bias
    for (int i = 0; i < output_size; ++i) {
        float activation = 0;
        for (int j = 0; j < input_size; ++j) {
            activation += weights[i][j] * input[j];
        }
        activation += biases[i];
        output[i] = activation;  // Apply ReLU activation
    }

    return output;
}
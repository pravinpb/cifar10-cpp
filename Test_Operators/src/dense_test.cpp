#include "dense.h"
#include <iostream>

int main() {
    Dense denseLayer(3, 2);  // 3 inputs, 2 outputs (neurons)

    // Example weights (2x3 matrix) and biases (2 elements)
    std::vector<std::vector<float>> weights = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    std::vector<float> biases = {0.1f, 0.2f};

    // Set weights and biases
    denseLayer.set_weights(weights);
    denseLayer.set_biases(biases);

    // Example input
    std::vector<float> input = {1.0f, 2.0f, 3.0f};

    // Get output from the dense layer
    std::vector<float> output = denseLayer.forward(input);

    // Print the output
    for (float value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

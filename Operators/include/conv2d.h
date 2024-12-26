#ifndef CONV2D_H
#define CONV2D_H

#include <vector>

// Conv2D class definition
class Conv2D {
public:
    Conv2D(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0);
    ~Conv2D();

    // Forward pass method
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
        const std::vector<float>& biases
    );

private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    // Helper function for padding
    std::vector<std::vector<std::vector<float>>> pad_input(
        const std::vector<std::vector<std::vector<float>>>& input, int padding);
};

#endif // CONV2D_H

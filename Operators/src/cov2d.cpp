#include "conv2d.h"
#include <vector>
#include <stdexcept>
#include <iostream>

// Constructor
Conv2D::Conv2D(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels_(input_channels), output_channels_(output_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

// Destructor
Conv2D::~Conv2D() {}

// Helper method for padding
std::vector<std::vector<std::vector<float>>> Conv2D::pad_input(
    const std::vector<std::vector<std::vector<float>>>& input, int padding) {
    if (padding == 0) return input;

    int padded_height = input[0].size() + 2 * padding;
    int padded_width = input[0][0].size() + 2 * padding;
    int channels = input.size();

    std::vector<std::vector<std::vector<float>>> padded_input(
        channels, std::vector<std::vector<float>>(padded_height, std::vector<float>(padded_width, 0.0f)));

    for (int c = 0; c < channels; ++c) {
        for (size_t h = 0; h < input[0].size(); ++h) {
            for (size_t w = 0; w < input[0][0].size(); ++w) {
                padded_input[c][h + padding][w + padding] = input[c][h][w];
            }
        }
    }
    return padded_input;
}

// Forward pass
std::vector<std::vector<std::vector<float>>> Conv2D::forward(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
    const std::vector<float>& biases) {
    
    std::cout << "Weights: " << weights.size() << " " << weights[0].size() << " " << weights[0][0].size() << " " << weights[0][0][0].size() << std::endl;
    std::cout << "Output channels: " << output_channels_ << std::endl;
    std::cout << "Input channels: " << input_channels_ << std::endl;
    if (weights.size() != output_channels_ || weights[0].size() != input_channels_) {
        throw std::invalid_argument("Weight dimensions do not match input or output channels.");
    }

    auto padded_input = pad_input(input, padding_);

    int input_height = padded_input[0].size();
    int input_width = padded_input[0][0].size();
    int output_height = (input_height - kernel_size_) / stride_ + 1;
    int output_width = (input_width - kernel_size_) / stride_ + 1;

    std::vector<std::vector<std::vector<float>>> output(
        output_channels_, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0f)));

    for (int oc = 0; oc < output_channels_; ++oc) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                float sum = biases[oc];
                for (int ic = 0; ic < input_channels_; ++ic) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int ih = oh * stride_ + kh;
                            int iw = ow * stride_ + kw;
                            sum += padded_input[ic][ih][iw] * weights[oc][ic][kh][kw];
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }

    return output;
}
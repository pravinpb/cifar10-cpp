#include "conv2d.h"

void conv2d(const std::vector<float>& input, const std::vector<float>& kernel,
            const std::vector<float>& bias, std::vector<float>& output,
            const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
            const std::array<int, 2>& kernel_size, const std::array<int, 2>& strides,
            const std::string& padding, const std::string& layer_name) {
    int batch = input_shape[0];
    int in_height = input_shape[1], in_width = input_shape[2], in_channels = input_shape[3];
    int out_height = output_shape[1], out_width = output_shape[2], out_channels = output_shape[3];
    int kernel_height = kernel_size[0], kernel_width = kernel_size[1];

    assert(input.size() == batch * in_height * in_width * in_channels);
    assert(kernel.size() == kernel_height * kernel_width * in_channels * out_channels);
    assert(bias.size() == out_channels);
    assert(output.size() == batch * out_height * out_width * out_channels);
    
    std::cout << "Performing Convolution operation for layer: " << layer_name << std::endl;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < out_channels; ++c) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int ih = h * strides[0] + kh;
                                int iw = w * strides[1] + kw;
                                if (padding == "same") {
                                    ih -= kernel_height / 2;
                                    iw -= kernel_width / 2;
                                }
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                                    int kernel_idx = ((kh * kernel_width + kw) * in_channels + ic) * out_channels + c;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((b * out_height + h) * out_width + w) * out_channels + c;
                    output[output_idx] = std::max(0.0f, sum + bias[c]);
                }
            }
        }
    }
}

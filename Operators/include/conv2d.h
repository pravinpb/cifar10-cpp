// Operators/conv2d.h
#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <cassert>
#include <iostream>

void conv2d(const std::vector<float>& input, const std::vector<float>& kernel,
            const std::vector<float>& bias, std::vector<float>& output,
            const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
            const std::array<int, 2>& kernel_size, const std::array<int, 2>& strides,
            const std::string& padding, const std::string& layer_name);

#endif
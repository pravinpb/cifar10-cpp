#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <array>
#include <string>
#include <cassert>
#include <iostream>
#include "relu.h"   // Assuming relu function is declared in relu.h
#include "softmax.h" // Assuming softmax function is declared in softmax.h

// Function declaration for the dense layer
void dense(const std::vector<float>& input, const std::vector<float>& weights,
           const std::vector<float>& bias, std::vector<float>& output,
           const std::array<int, 2>& input_shape, const std::array<int, 2>& output_shape,
           const std::string& activation, const std::string layer_name);

#endif // DENSE_H

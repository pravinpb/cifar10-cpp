#include "modeltest.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>

// Function to read a file into a vector of floats
std::vector<float> read_output_file(const std::string& file_path) {
    std::vector<float> output;
    std::ifstream file(file_path);

    if (!file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return output;
    }

    double value;
    while (file >> value) {
        output.push_back(static_cast<float>(value));
    }
    return output;
}

// Function to compare predicted labels
void compare_model_outputs(const std::string& cpp_file, const std::string& python_file) {
    // Read the output files
    std::vector<float> cpp_output = read_output_file(cpp_file);
    std::vector<float> python_output = read_output_file(python_file);

    if (cpp_output.empty() || python_output.empty()) {
        std::cerr << "Error: One or both files are empty or failed to load." << std::endl;
        return;
    }

    // Find the indices of the maximum values
    auto cpp_max_it = std::max_element(cpp_output.begin(), cpp_output.end());
    auto python_max_it = std::max_element(python_output.begin(), python_output.end());

    int cpp_label = std::distance(cpp_output.begin(), cpp_max_it);
    int python_label = std::distance(python_output.begin(), python_max_it);

    // Compare the labels
    if (cpp_label == python_label) {
        std::cout << "Model test passed: Predicted class is the same (" << cpp_label << ")" << " ";
    } else {
        std::cout << "Model test failed: C++ predicted class " << cpp_label
                  << ", Python predicted class " << python_label << " ";
    }
}

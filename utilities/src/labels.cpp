#include "labels.h"
#include <iostream>
#include <algorithm>

// Function to predict and print the class
void predict_and_print_class(const std::vector<float>& input_data) {
    // List of class names
    std::vector<std::string> class_names = {
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };

    // Find the index of the maximum probability
    auto max_prob_it = std::max_element(input_data.begin(), input_data.end());
    int max_prob_index = std::distance(input_data.begin(), max_prob_it);

    // Print the predicted class
    std::cout << class_names[max_prob_index] << std::endl;
}

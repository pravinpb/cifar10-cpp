#include <iostream>
#include <fstream>  // For file operations
#include "conv2d.h"
#include "maxpooling.h"
#include "dense.h"
#include "loadjsonfile.h"
#include "loadbinfile.h"
#include "relu.h"
#include "softmax.h"


int main() {
    std::string config_path = "/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/toTest/data/model_architecture.json";
    nlohmann::json config = load_json_config(config_path);

    std::vector<float> input_data(32 * 32 * 3, 1.0f);

    // //print input
    // std::cout << "input" << std::endl;
    // for (float val : input_data) {
    //     std::cout << val << " ";
    // }

    for (const auto& layer : config["layers"]) {
        std::string layer_type = layer["type"];
        if (layer_type == "Conv2D") {
            std::vector<float> kernel = load_binary_file(layer["weights_file_paths"][0]);
            std::vector<float> bias = load_binary_file(layer["weights_file_paths"][1]);
            std::array<int, 4> input_shape = {1, layer["attributes"]["input_shape"][0], layer["attributes"]["input_shape"][1], layer["attributes"]["input_shape"][2]};
            std::array<int, 4> output_shape = {1, layer["attributes"]["output_shape"][0], layer["attributes"]["output_shape"][1], layer["attributes"]["output_shape"][2]};
            std::array<int, 2> kernel_size = {layer["attributes"]["kernel_size"][0], layer["attributes"]["kernel_size"][1]};
            std::array<int, 2> strides = {layer["attributes"]["strides"][0], layer["attributes"]["strides"][1]};
            std::string padding = layer["attributes"]["padding"];
            std::string activation = layer["attributes"]["activation"];

            std::vector<float> conv_output(output_shape[1] * output_shape[2] * output_shape[3]);
            conv2d(input_data, kernel, bias, conv_output, input_shape, output_shape, kernel_size, strides, padding, layer["layer_name"]);

            if (activation == "relu") {
                relu(conv_output);
            }
            input_data = conv_output;

            // //print convolution output
            // std::cout << "convolution Output: " << std::endl;
            // for (float val : input_data) {
            //     std::cout << val << " ";
            // }
            // std::cout << std::endl;
            
        }
        else if (layer_type == "MaxPooling2D") {
            std::array<int, 4> input_shape = {1, layer["attributes"]["input_shape"][0], layer["attributes"]["input_shape"][1], layer["attributes"]["input_shape"][2]};
            std::array<int, 4> output_shape = {1, layer["attributes"]["output_shape"][0], layer["attributes"]["output_shape"][1], layer["attributes"]["output_shape"][2]};
            std::array<int, 2> pool_size = {2, 2};
            std::array<int, 2> strides = {layer["attributes"]["strides"][0], layer["attributes"]["strides"][1]};
            std::string padding = layer["attributes"]["padding"];
            std::string layer_name = layer["layer_name"];

            std::vector<float> maxpool_output(output_shape[1] * output_shape[2] * output_shape[3]);
            max_pooling2d(input_data, maxpool_output, input_shape, output_shape, pool_size, strides, padding, layer_name);

            input_data = maxpool_output;

            // //print maxpool output
            // std::cout << "MaxPooling Output: " << std::endl;
            // for (float val : input_data) {
            //     std::cout << val << " ";
            // }
            // std::cout << std::endl;
        }
        else if (layer_type == "Dense") {
            std::vector<float> weights = load_binary_file(layer["weights_file_paths"][0]);
            std::vector<float> bias = load_binary_file(layer["weights_file_paths"][1]);
            std::array<int, 2> input_shape = {1, layer["attributes"]["input_shape"][0]};
            std::array<int, 2> output_shape = {1, layer["attributes"]["output_shape"][0]};
            std::string activation = layer["attributes"]["activation"];
            std::string layer_name = layer["layer_name"];

            std::vector<float> dense_output(output_shape[1]);
            
            dense(input_data, weights, bias, dense_output, input_shape, output_shape, activation, layer_name);

            input_data = dense_output;


        //     //print dense output
        //     std::cout << "Dense Output: " << std::endl;
        //     for (float val : input_data) {
        //         std::cout << val << " ";
        //     }
        //     std::cout << std::endl;
        }
    }
    std::cout << " " << std::endl;
    std::cout << "Final output size: " << input_data.size() << std::endl;
    for (float val : input_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout <<" "<< std::endl;

    std::vector<std::string> class_names = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    auto max_prob_it = std::max_element(input_data.begin(), input_data.end());
    int max_prob_index = std::distance(input_data.begin(), max_prob_it);
    std::cout << "Predicted class: " << class_names[max_prob_index] << std::endl;
    return 0;
}

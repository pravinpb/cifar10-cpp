#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <cmath> // For softmax activation
#include "conv2d.h"        // Conv2D layer implementation
#include "maxpooling.h"    // MaxPooling2D layer implementation
#include "flatten.h"       // Flatten layer implementation
#include "dense.h"         // Dense layer implementation

using json = nlohmann::json; // Alias for nlohmann::json

// Function to read the JSON configuration
json read_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening JSON file.");
    }

    json j;
    file >> j;
    return j;
}

// Function to load the weights from binary files (for Conv2D and Dense)
std::vector<std::vector<std::vector<std::vector<float>>>> load_weights(const std::string& file_path, int output_channels, int input_channels, int kernel_size) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening weights file: " + file_path);
    }

    std::vector<std::vector<std::vector<std::vector<float>>>> weights(output_channels,
        std::vector<std::vector<std::vector<float>>>(input_channels,
            std::vector<std::vector<float>>(kernel_size,
                std::vector<float>(kernel_size))));

    for (int c_out = 0; c_out < output_channels; ++c_out) {
        for (int c_in = 0; c_in < input_channels; ++c_in) {
            for (int h = 0; h < kernel_size; ++h) {
                for (int w = 0; w < kernel_size; ++w) {
                    file.read(reinterpret_cast<char*>(&weights[c_out][c_in][h][w]), sizeof(float));
                }
            }
        }
    }

    file.close();
    return weights;
}

// Function to load biases from binary files (for Conv2D and Dense)
std::vector<float> load_biases(const std::string& file_path, int output_channels) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening biases file: " + file_path);
    }

    std::vector<float> biases(output_channels);
    file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));

    file.close();
    return biases;
}

// Function to create a hardcoded 32x32x3 image (RGB values set manually)
std::vector<std::vector<std::vector<float>>> create_hardcoded_image(int height, int width, int channels) {
    std::vector<std::vector<std::vector<float>>> image(channels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

    // Set some fixed values for the image (simple pattern for demonstration)
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Example: Just a simple pattern
                image[c][h][w] = static_cast<float>((h + w + c) % 256) / 255.0f;  // Normalize to [0, 1]
            }
        }
    }

    return image;
}

// Apply ReLU activation function
void apply_relu(std::vector<std::vector<std::vector<float>>>& data) {
    for (auto& channel : data) {
        for (auto& row : channel) {
            for (auto& val : row) {
                val = std::max(val, 0.0f);
            }
        }
    }
}

// Apply Softmax activation function (for the final Dense layer)
void apply_softmax(std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += std::exp(val);
    }
    for (float& val : data) {
        val = std::exp(val) / sum;
    }
}

int main() {
    try {
        // Load the JSON configuration file
        std::string json_file = "/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/configs/json/model_config.json";
        std::cout << "Loading JSON configuration from: " << json_file << std::endl;
        json config = read_json(json_file);


        // Iterate over each layer in the model
        std::vector<std::vector<std::vector<std::vector<float>>>> input_data;
        std::vector<float> output_data;

        for (const auto& layer : config["layers"]) {
            std::string layer_name = layer["layer_name"];
            std::string layer_type = layer["type"];

            // Load layer attributes
            auto attributes = layer["attributes"];


            if (layer.contains("weights_file_paths")) {

                if (layer["weights_file_paths"].is_null()) {
                    std::cerr << "Warning: weights_file_paths is null for layer " << layer_name << ". Skipping weight loading." << std::endl;
                } else if (layer["weights_file_paths"].is_array()) {
                    const auto& weights_file_paths = layer["weights_file_paths"];
                    std::cout << "Found weights_file_paths for layer: " << layer_name << std::endl;
                    std::cout << "Number of weights paths: " << weights_file_paths.size() << std::endl;

                    if (weights_file_paths.size() < 2) {
                        throw std::runtime_error("Error: Weights file paths are incomplete for layer " + layer_name);
                    }

                    std::cout << "First weights file path: " << weights_file_paths[0] << std::endl;
                    std::cout << "Second weights file path: " << weights_file_paths[1] << std::endl;
                } else {
                    std::cerr << "Warning: weights_file_paths is not an array for layer " << layer_name << ". Skipping weight loading." << std::endl;
                }
            } else {
                std::cerr << "Warning: No weights_file_paths found for layer " << layer_name << ". Skipping weight loading." << std::endl;
            }


            // Load weights and biases (if applicable)
            std::vector<std::vector<std::vector<std::vector<float>>>> kernel;
            std::vector<float> bias;

            if ((layer_type == "Conv2D" || layer_type == "Dense") &&
                layer.contains("weights_file_paths") && layer["weights_file_paths"].is_array()) {

                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];
                std::vector<int> kernel_size = attributes["kernel_size"];

                if (layer_type == "Conv2D") {
                    std::cout << "Loading weights and biases for Conv2D" << std::endl;
                    const auto& weights_file_paths = layer["weights_file_paths"];
                    kernel = load_weights(weights_file_paths[0], output_shape[2], input_shape[2], kernel_size[0]);
                    bias = load_biases(weights_file_paths[1], output_shape[2]);
                } else if (layer_type == "Dense") {
                    const auto& weights_file_paths = layer["weights_file_paths"];
                    kernel = load_weights(weights_file_paths[0], output_shape[0], input_shape[0], 1);
                    bias = load_biases(weights_file_paths[1], output_shape[0]);
                }
            }

            // Hardcoded input data for the first layer
            if (layer_name == "conv2d") {
                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];
                std::cout << "Creating hardcoded image for Conv2D" << std::endl;
                input_data = {create_hardcoded_image(input_shape[0], input_shape[1], input_shape[2])};
                std::cout << "Input size for " << layer_name << ": (" << input_shape[2] << ", " << input_shape[0] << ", " << input_shape[1] << ")\n";
            }

            // Process layer based on type
            if (layer_type == "Conv2D") {

                std::cout << "Input data shape: (" << input_data.size() << ", "
                    << input_data[0].size() << ", "
                    << input_data[0][0].size() << ")\n";

                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];
                std::vector<int> kernel_size = attributes["kernel_size"];
                std::vector<int> strides = attributes["strides"];
                std::string padding = attributes["padding"];
                std::string activation = attributes["activation"];

                std::cout << "Layer name: " << layer_name << std::endl;

                // Create Conv2D layer and apply
                Conv2D conv(input_shape[2], output_shape[2], kernel_size[0], strides[0], (padding == "valid") ? 0 : 1);
                std::cout << "Forwarding Conv2D with input data..." << std::endl;
                auto conv_output = conv.forward(input_data[0], kernel, bias);

                // Apply activation function
                if (activation == "relu") {
                    std::cout << "Applying ReLU activation" << std::endl;
                    apply_relu(conv_output);
                }

                // Print output size
                std::cout << "Output size for " << layer_name << ": (" << output_shape[2] << ", " << output_shape[0] << ", " << output_shape[1] << ")\n";
            } 
            else if (layer_type == "MaxPooling2D") {

                std::cout << "Input data shape: (" << input_data.size() << ", "
                    << input_data[0].size() << ", "
                    << input_data[0][0].size() << ")\n";

                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];
                std::vector<int> strides = attributes["strides"];
                std::string padding = attributes["padding"];

                std::cout << "Layer name: " << layer_name << std::endl;

                MaxPooling pool(2, 1);
                std::cout << "Forwarding MaxPooling2D..." << std::endl;
                auto pool_output = pool.apply_pooling(input_data[0][0]);

                // Print output size
                std::cout << "Output size for " << layer_name << ": (" << pool_output.size() << ", " << pool_output[0].size() << ")\n";
            }
            else if (layer_type == "Flatten") {

                std::cout << "Input data shape: (" << input_data.size() << ", "
                    << input_data[0].size() << ", "
                    << input_data[0][0].size() << ")\n";

                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];

                std::cout << "Layer name: " << layer_name << std::endl;

                Flatten flatten;
                std::cout << "Flattening output..." << std::endl;
                auto flatten_output = flatten.forward(input_data[0][0]);

                // Print output size
                std::cout << "Flatten output size: (" << flatten_output.size() << ")\n";
            } 
            else if (layer_type == "Dense") {

                std::cout << "Input data shape: (" << input_data.size() << ", "
                    << input_data[0].size() << ", "
                    << input_data[0][0].size() << ")\n";

                std::vector<int> input_shape = attributes["input_shape"];
                std::vector<int> output_shape = attributes["output_shape"];
                std::string activation = attributes["activation"];

                std::cout << "Layer name: " << layer_name << std::endl;

                Dense dense(input_shape[0], output_shape[0]);
                std::cout << "Forwarding Dense layer..." << std::endl;
                Flatten flatten;
                auto flattened_input = flatten.forward(input_data[0][0]);
                output_data = dense.forward(flattened_input);

                // Apply activation
                if (activation == "softmax") {
                    std::cout << "Applying Softmax activation" << std::endl;
                    apply_softmax(output_data);
                }

                // Print output size
                std::cout << "Output size for " << layer_name << ": (" << output_data.size() << ")\n";
            }
        }
    
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    return 0;
}
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

using json = nlohmann::json;

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

// Function to load weights from binary files
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

//Function to load weights for dense layers
std::vector<std::vector<float>> load_weights_dense(const std::string& file_path, int output_channels, int input_channels) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening weights file: " + file_path);
    }
    std::vector<std::vector<float>> weights(output_channels, std::vector<float>(input_channels));
    for (int i = 0; i < output_channels; ++i) {
        file.read(reinterpret_cast<char*>(weights[i].data()), weights[i].size() * sizeof(float));
    }
    file.close();
    return weights;
}

// Function to load biases from binary files
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

// Function to create a hardcoded 32x32x3 image (for the first layer input)
std::vector<std::vector<std::vector<float>>> create_hardcoded_image(int height, int width, int channels) {
    std::vector<std::vector<std::vector<float>>> image(channels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                image[c][h][w] = static_cast<float>((h + w + c) % 256) / 255.0f;  // Normalized [0, 1]
            }
        }
    }
    return image;
}

// Apply ReLU activation
void apply_relu(std::vector<std::vector<std::vector<float>>>& data) {
    for (auto& channel : data) {
        for (auto& row : channel) {
            for (auto& val : row) {
                val = std::max(val, 0.0f);
            }
        }
    }
}

// Apply ReLU activation (for Dense layers)
void apply_relu(std::vector<float>& data) {
    for (float& val : data) {
        val = std::max(val, 0.0f);
    }
}

// Apply Softmax activation (for Dense layers)
void apply_softmax(std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += std::exp(val);
    }
    for (float& val : data) {
        val = std::exp(val) / sum;
    }
}

// Function to print all data to both console and file
void print_all_data(const std::vector<std::vector<std::vector<float>>>& data, std::ofstream& file) {
    file << "Output data: \n";
    std::cout << "Output data: \n";
    for (const auto& channel : data) {
        for (const auto& row : channel) {
            for (float val : row) {
                file << val << " ";
                std::cout << val << " ";
            }
            file << "\n";
            std::cout << "\n";
        }
        file << "\n";
        std::cout << "\n";
    }
    file << std::endl;
    std::cout << std::endl;
}

void print_all_data(const std::vector<float>& data, std::ofstream& file) {
    file << "Output data: \n";
    std::cout << "Output data: \n";
    for (float val : data) {
        file << val << " ";
        std::cout << val << " ";
    }
    file << std::endl;
    std::cout << std::endl;
}

int main() {
    try {
        // Load the JSON configuration file
        std::string json_file = "/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/configs/json/model_config.json";
        json config = read_json(json_file);

        // Global variables for input/output
        std::vector<std::vector<std::vector<float>>> current_output_data;
        std::vector<int> current_output_shape;

        // Process each layer in the model
        for (const auto& layer : config["layers"]) {
            std::string layer_name = layer["layer_name"];
            std::string layer_type = layer["type"];
            auto attributes = layer["attributes"];

            // Initialize weights and biases if applicable
            std::vector<std::vector<std::vector<std::vector<float>>>> kernel;
            std::vector<std::vector<float>> kernel_dense;
            std::vector<float> bias;

            if (layer_type == "Conv2D" && layer.contains("weights_file_paths")) {
                const auto& weights_file_paths = layer["weights_file_paths"];
                kernel = load_weights(weights_file_paths[0], attributes["output_shape"][2], attributes["input_shape"][2], attributes["kernel_size"][0]);
                bias = load_biases(weights_file_paths[1], attributes["output_shape"][2]);
            }

            if (layer_name == "conv2d" && current_output_data.empty()) {
                current_output_data = create_hardcoded_image(attributes["input_shape"][0], attributes["input_shape"][1], attributes["input_shape"][2]);
                current_output_shape = {attributes["input_shape"][0], attributes["input_shape"][1], attributes["input_shape"][2]};
            }

            // Print layer name and shapes
            std::cout << "\nProcessing Layer: " << layer_name << " (" << layer_type << ")" << std::endl;
            std::cout << "Input shape: ";
            for (const auto& dim : current_output_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            if (layer_type == "Conv2D") {
                Conv2D conv(current_output_shape[2], attributes["output_shape"][2], attributes["kernel_size"][0], attributes["strides"][0], (attributes["padding"] == "valid") ? 0 : 1);
                auto conv_output = conv.forward(current_output_data, kernel, bias);

                if (attributes["activation"] == "relu") {
                    apply_relu(conv_output);
                }

                current_output_data = conv_output;
                current_output_shape = {attributes["output_shape"][0], attributes["output_shape"][1], attributes["output_shape"][2]};
            } else if (layer_type == "MaxPooling2D") {
                int pool_size = 2;  // Default pool size of 2x2
                int stride = attributes["strides"][0];
                MaxPooling pool(pool_size, stride);
                auto pool_output = pool.apply_pooling(current_output_data);

                current_output_data = {pool_output};
                current_output_shape = {static_cast<int>(pool_output[0][0].size()), static_cast<int>(pool_output[0].size()), static_cast<int>(pool_output.size())};
            } else if (layer_type == "Flatten") {
                Flatten flatten;
                auto flatten_output = flatten.forward(current_output_data);

                current_output_data = {{flatten_output}}; // Wrap in an extra dimension
                current_output_shape = {static_cast<int>(flatten_output.size())};
            } else if (layer_type == "Dense") {
                Dense dense(current_output_shape[0], attributes["output_shape"][0]);
                std::vector<float> dense_output;

                if (layer_type == "Dense" && layer.contains("weights_file_paths")) {
                    const auto& weights_file_paths = layer["weights_file_paths"];
                    kernel_dense = load_weights_dense(weights_file_paths[0], attributes["output_shape"][0], current_output_shape[0]);
                    bias = load_biases(weights_file_paths[1], attributes["output_shape"][0]);
                }

                dense.set_weights(kernel_dense);
                dense.set_biases(bias);
                
                dense_output = dense.forward(current_output_data[0][0]);

                if (attributes["activation"] == "softmax") {
                    apply_softmax(dense_output);
                } else if (attributes["activation"] == "relu") {
                    apply_relu(dense_output);
                }

                current_output_data = {{dense_output}};  // Wrap in an extra dimension
                current_output_shape = {static_cast<int>(dense_output.size())};
            }

            // Print output shape
            std::cout << "Output shape: ";
            for (const auto& dim : current_output_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }
        
        // Print the final output data
        std::cout << "Dense output: ";
        for (const auto& val : current_output_data[0][0]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}


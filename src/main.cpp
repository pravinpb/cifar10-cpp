#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>  // Include the nlohmann/json header for JSON parsing
#include "conv2d.h"
#include "maxpooling.h"
#include "flatten.h"
#include "dense.h"

using json = nlohmann::json;

// Function to read binary data from a file into a vector
template <typename T>
std::vector<T> read_binary_file(const std::string& file_path, size_t size) {
    std::ifstream file(file_path, std::ios::binary);
    std::vector<T> data(size);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    } else {
        std::cerr << "Error reading file: " << file_path << std::endl;
    }
    return data;
}

// Function to write binary data to a file
template <typename T>
void write_binary_file(const std::string& file_path, const std::vector<T>& data) {
    std::ofstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    } else {
        std::cerr << "Error writing file: " << file_path << std::endl;
    }
}

// Function to reshape a 1D vector into a 3D vector (for Conv2D input)
std::vector<std::vector<std::vector<float>>> reshape_to_3d(
    const std::vector<float>& data, int height, int width, int channels) {
    std::vector<std::vector<std::vector<float>>> reshaped_data(
        channels, std::vector<std::vector<float>>(height, std::vector<float>(width)));

    int index = 0;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                reshaped_data[c][h][w] = data[index++];
            }
        }
    }
    return reshaped_data;
}

// Function to reshape a 1D vector into a 4D vector (for Conv2D weights)
std::vector<std::vector<std::vector<std::vector<float>>>> reshape_to_4d_weights(
    const std::vector<float>& data, int input_channels, int output_channels, int kernel_size) {
    std::vector<std::vector<std::vector<std::vector<float>>>> reshaped_weights(
        output_channels, std::vector<std::vector<std::vector<float>>>(
                             input_channels, std::vector<std::vector<float>>(
                                                kernel_size, std::vector<float>(kernel_size))));

    int index = 0;
    for (int o = 0; o < output_channels; ++o) {
        for (int i = 0; i < input_channels; ++i) {
            for (int h = 0; h < kernel_size; ++h) {
                for (int w = 0; w < kernel_size; ++w) {
                    reshaped_weights[o][i][h][w] = data[index++];
                }
            }
        }
    }
    return reshaped_weights;
}

int main() {
    // Load the JSON configuration from a file
    std::ifstream file("/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/configs/json/model_config.json");
    if (!file.is_open()) {
        std::cerr << "Error opening config file!" << std::endl;
        return 1;
    }

    // Parse the JSON file
    json config;
    file >> config;
    std::cout << "Loaded config: " << config.dump(4) << std::endl;  // Print the parsed JSON config

    // Variables to hold the layer outputs
    std::vector<float> inputData;
    std::vector<std::vector<std::vector<float>>> convOutput;
    std::vector<float> poolOutput, flattenOutput, denseOutput;

    // Iterate through the layers in the configuration
    for (const auto& layerConfig : config["layers"]) {
        std::string layerType = layerConfig["type"];
        std::string inputFilePath = layerConfig["input_file_path"];
        std::string outputFilePath = layerConfig["output_file_path"];
        std::vector<std::string> weightFilePaths = layerConfig["weights_file_paths"];

        std::cout << "Processing layer: " << layerType << std::endl;
        std::cout << "Input file: " << inputFilePath << std::endl;
        std::cout << "Output file: " << outputFilePath << std::endl;

        // Read input data for this layer
        size_t inputSize = 0;  // Determine the input size from the layer attributes (e.g., input_shape)
        if (!layerConfig["attributes"]["input_shape"].is_null()) {
            inputSize = layerConfig["attributes"]["input_shape"].size(); // Simplified; adjust as needed
        }

        std::cout << "Reading input data of size: " << inputSize << std::endl;
        inputData = read_binary_file<float>(inputFilePath, inputSize);  // Read the input data from file

        if (layerType == "Conv2D") {
            // Extract attributes
            int input_channels = layerConfig["attributes"]["input_shape"][3]; // Assuming last element is input channels
            int output_channels = layerConfig["attributes"]["output_shape"][3]; // Assuming last element is output channels
            int kernelHeight = layerConfig["attributes"]["kernel_size"][0];
            int kernelWidth = layerConfig["attributes"]["kernel_size"][1];
            int stride = layerConfig["attributes"]["strides"][0];
            std::string padding = layerConfig["attributes"]["padding"];
            std::string activation = layerConfig["attributes"]["activation"];

            std::cout << "Conv2D layer: input_channels = " << input_channels
                      << ", output_channels = " << output_channels
                      << ", kernel_size = (" << kernelHeight << ", " << kernelWidth << ")"
                      << ", stride = " << stride
                      << ", padding = " << padding
                      << ", activation = " << activation << std::endl;

            // Read weights (kernel and bias)
            int kernelSize = input_channels * output_channels * kernelHeight * kernelWidth;
            std::cout << "Reading kernel weights of size: " << kernelSize << std::endl;
            std::vector<float> kernel = read_binary_file<float>(weightFilePaths[0], kernelSize);
            std::cout << "Reading bias weights of size: " << output_channels << std::endl;
            std::vector<float> bias = read_binary_file<float>(weightFilePaths[1], output_channels);

            // Reshape the input data to match the expected format (3D vector for Conv2D)
            int height = layerConfig["attributes"]["input_shape"][1];
            int width = layerConfig["attributes"]["input_shape"][2];
            auto reshapedInput = reshape_to_3d(inputData, height, width, input_channels);

            std::cout << "Input reshaped to 3D: [" << input_channels << " x " << height << " x " << width << "]" << std::endl;

            // Reshape the weights to match the expected format (4D vector for Conv2D)
            auto reshapedWeights = reshape_to_4d_weights(kernel, input_channels, output_channels, kernelHeight);

            // Create the Conv2D layer and apply it
            Conv2D convLayer(input_channels, output_channels, kernelHeight, stride, (padding == "valid") ? 0 : 1);
            std::cout << "Applying Conv2D forward pass..." << std::endl;
            convOutput = convLayer.forward(reshapedInput, reshapedWeights, bias);

            std::cout << "Conv2D forward pass completed." << std::endl;

            // Apply activation function if needed (e.g., ReLU)
            if (activation == "relu") {
                std::cout << "Applying ReLU activation..." << std::endl;
                for (auto& channel : convOutput) {
                    for (auto& row : channel) {
                        for (auto& value : row) {
                            value = std::max(static_cast<float>(0.0f), value);  // ReLU activation
                        }
                    }
                }
                std::cout << "ReLU activation applied." << std::endl;
            }

            write_binary_file(outputFilePath, convOutput);  // Write the output to file
            std::cout << "Conv2D output written to: " << outputFilePath << std::endl;
        }
        else if (layerType == "MaxPooling2D") {
            // Handle MaxPooling2D layer (same as before)
            std::cout << "MaxPooling2D layer processing..." << std::endl;
        }
        else if (layerType == "Flatten") {
            // Handle Flatten layer (same as before)
            std::cout << "Flatten layer processing..." << std::endl;
        }
        else if (layerType == "Dense") {
            // Handle Dense layer (same as before)
            std::cout << "Dense layer processing..." << std::endl;
        }
    }

    return 0;
}

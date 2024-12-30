#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std;

// Function to read a binary file into a vector
template<typename T>
vector<T> read_binary_file(const string& file_path) {
    ifstream file(file_path, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + file_path);
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);

    vector<T> buffer(file_size / sizeof(T));  // Adjust size for type
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    return buffer;
}

// Function to write a vector to the output file (for debugging)
template<typename T>
void write_vector(ofstream& output_file, const vector<T>& vec) {
    for (const auto& val : vec) {
        output_file << val << " ";
    }
    output_file << endl;
}

// Function to create a proper file path using filesystem::path
fs::path get_full_path(const fs::path& base_dir, const string& file_name) {
    return base_dir / file_name;  // Use filesystem path concatenation
}

int main() {
    // Set base path for the data directory
    fs::path data_dir = "/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/data/";

    // Open the output text file
    ofstream output_file("/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/toTest/results.txt");
    if (!output_file.is_open()) {
        cerr << "Failed to open output file!" << endl;
        return 1;
    }

    // Open the JSON file
    ifstream input_file("/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/configs/json/model_config.json");
    
    if (!input_file.is_open()) {
        cerr << "Failed to open JSON file!" << endl;
        return 1;
    }

    // Parse the JSON file
    json model;
    input_file >> model;

    // Iterate through layers and process each layer
    for (const auto& layer : model["layers"]) {
        output_file << "Layer: " << layer["layer_name"] << endl;
        output_file << "  Type: " << layer["type"] << endl;
        
        // Write and process input shape
        auto input_shape = layer["attributes"]["input_shape"];
        output_file << "  Input Shape: ";
        for (const auto& dim : input_shape) {
            output_file << dim << " ";
        }
        output_file << endl;

        // Write and process output shape
        auto output_shape = layer["attributes"]["output_shape"];
        output_file << "  Output Shape: ";
        for (const auto& dim : output_shape) {
            output_file << dim << " ";
        }
        output_file << endl;

        // Build the file path for the input binary file based on the layer name
        fs::path input_file_path = get_full_path(data_dir / "input", layer["layer_name"].get<string>() + "_input.bin");
        output_file << "  Reading input from: " << input_file_path << endl;
        try {
            auto input_data = read_binary_file<float>(input_file_path.string());  // Convert to string for file reading
            output_file << "    Input Data (first 10 elements): ";
            write_vector(output_file, vector<float>(input_data.begin(), input_data.begin() + min(input_data.size(), size_t(10))));
        } catch (const exception& e) {
            output_file << "    Error reading input data: " << e.what() << endl;
        }

        // Reading weights binary files (if any)
        auto weights_file_paths = layer["weights_file_paths"];
        for (const auto& weights_file_path : weights_file_paths) {
            // Make sure we are appending the weights file path correctly relative to the "weights" directory
            fs::path weight_file_path = get_full_path(data_dir / "weights", weights_file_path.get<string>());
            output_file << "  Reading weights from: " << weight_file_path << endl;
            try {
                auto weights_data = read_binary_file<float>(weight_file_path.string());  // Convert to string for file reading
                output_file << "    Weights Data (first 10 elements): ";
                write_vector(output_file, vector<float>(weights_data.begin(), weights_data.begin() + min(weights_data.size(), size_t(10))));
            } catch (const exception& e) {
                output_file << "    Error reading weights data: " << e.what() << endl;
            }
        }

        // Build the file path for the output binary file based on the layer name
        fs::path output_file_path = get_full_path(data_dir / "output", layer["layer_name"].get<string>() + "_output.bin");
        output_file << "  Reading output from: " << output_file_path << endl;
        try {
            auto output_data = read_binary_file<float>(output_file_path.string());  // Convert to string for file reading
            output_file << "    Output Data (first 10 elements): ";
            write_vector(output_file, vector<float>(output_data.begin(), output_data.begin() + min(output_data.size(), size_t(10))));
        } catch (const exception& e) {
            output_file << "    Error reading output data: " << e.what() << endl;
        }

        output_file << endl;  // Add a new line between layers
    }

    // Close the output file
    output_file.close();

    cout << "Results have been written to results.txt." << endl;
    return 0;
}

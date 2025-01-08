#include "saveastxt.h"
#include <filesystem>

void save_output_as_txt(const std::string& layer_name, const std::vector<float>& data) {
    // Ensure the directory exists
    std::filesystem::create_directories("/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Output/cpp");  // Creates the directory if it doesn't exist
    
    // Construct file path using the layer name
    std::string file_path = "/Users/pravinpb/pycode/MCW/Assignments/cifar10-cpp/Output/cpp/" + layer_name + "_output.txt";
    
    // Open file for writing (this will create the file if it doesn't exist)
    std::ofstream out_file(file_path);
    if (!out_file) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    // Write data to the file
    for (float val : data) {
        out_file << val << " ";
    }

    out_file << std::endl;  // Add a newline after writing all values
    out_file.close();
}

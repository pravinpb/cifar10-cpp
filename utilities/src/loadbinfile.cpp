#include <nlohmann/json.hpp>
#include "loadbinfile.h"



nlohmann::json load_json_config(const std::string& file_path) {
    std::ifstream config_file(file_path);
    if (!config_file) {
        std::cerr << "Error opening config file." << std::endl;
        exit(1);
    }
    nlohmann::json config;
    config_file >> config;
    return config;
}
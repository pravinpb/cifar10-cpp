#ifndef LOADJSONFILE_H
#define LOADJSONFILE_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

nlohmann::json load_json_config(const std::string& file_path);

#endif
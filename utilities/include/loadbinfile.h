#ifndef LOADBINFILE_H
#define LOADBINFILE_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

std::vector<float> load_binary_file(const std::string& file_path);

#endif
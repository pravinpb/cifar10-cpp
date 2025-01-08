#ifndef SAVEASTXT_H
#define SAVEASTXT_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

void save_output_as_txt(const std::string& layer_name, const std::vector<float>& data);

#endif // SAVEASTXT_H

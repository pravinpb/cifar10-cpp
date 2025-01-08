#include "unittest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cctype> // For std::isspace

namespace fs = std::filesystem;

// Function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    size_t last = str.find_last_not_of(" \n\r\t");
    return (first == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

// Function to read a file into a vector of strings
std::vector<std::string> read_file(const std::string& file_path) {
    std::vector<std::string> content;
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return content;
    }
    std::string line;
    while (std::getline(file, line)) {
        content.push_back(trim(line)); // Normalize the line
    }
    return content;
}

// Function to compare two floating-point numbers with a tolerance
bool is_equal(double a, double b, double tolerance) {
    return std::fabs(a - b) <= tolerance;
}

// Function to compute the match percentage between two files
double compare_files(const std::string& file1, const std::string& file2) {
    std::vector<std::string> content1 = read_file(file1);
    std::vector<std::string> content2 = read_file(file2);

    size_t total_lines = std::max(content1.size(), content2.size());
    if (total_lines == 0) return 100.0; // Both files are empty, so they're 100% similar

    size_t match_count = 0;

    for (size_t i = 0; i < total_lines; ++i) {
        if (i < content1.size() && i < content2.size()) {
            std::istringstream stream1(content1[i]);
            std::istringstream stream2(content2[i]);
            
            double num1, num2;
            bool match = true;

            while (stream1 >> num1 && stream2 >> num2) {
                if (!is_equal(num1, num2)) {
                    match = false;
                    break;
                }
            }

            // Ensure both streams reached the end (equal number of tokens)
            if (match && stream1.eof() && stream2.eof()) {
                ++match_count;
            }
        }
    }

    return (static_cast<double>(match_count) / total_lines) * 100.0;
}

// Function to compare files in two directories
void compare_directories(const std::string& dir1, const std::string& dir2) {
    // Collect all files in the directories
    std::vector<std::string> files_dir1;
    std::vector<std::string> files_dir2;

    for (const auto& entry : fs::directory_iterator(dir1)) {
        if (entry.is_regular_file()) {
            files_dir1.push_back(entry.path().filename().string());
        }
    }

    for (const auto& entry : fs::directory_iterator(dir2)) {
        if (entry.is_regular_file()) {
            files_dir2.push_back(entry.path().filename().string());
        }
    }

    // Compare files with the same name
    for (const auto& file1 : files_dir1) {
        auto it = std::find(files_dir2.begin(), files_dir2.end(), file1);
        if (it != files_dir2.end()) {
            std::string file_path1 = dir1 + "/" + file1;
            std::string file_path2 = dir2 + "/" + *it;

            double match_percentage = compare_files(file_path1, file_path2);
            std::cout << "Match for " << file1 << ": " << match_percentage << "%\n";
        } else {
            std::cout << "File " << file1 << " not found in " << dir2 << "\n";
        }
    }
}

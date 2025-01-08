#ifndef UNITTEST_H
#define UNITTEST_H

#include <string>
#include <vector>

// Function declarations
std::vector<std::string> read_file(const std::string& file_path);
bool is_equal(double a, double b, double tolerance = 1e-4);
double compare_files(const std::string& file1, const std::string& file2);
void compare_directories(const std::string& dir1, const std::string& dir2);

#endif // UNITTEST_H

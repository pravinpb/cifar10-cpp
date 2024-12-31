
#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

class Flatten {
public:
    Flatten() = default;
    
    std::vector<float> forward(const std::vector<std::vector<float>>& input);

private:
    std::vector<float> flatten_2d(const std::vector<std::vector<float>>& input);
};

#endif

#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include <vector>

class MaxPooling {
public:
    MaxPooling(int pool_size, int stride);
    
    
std::vector<std::vector<std::vector<float>>> apply_pooling(const std::vector<std::vector<std::vector<float>>>& input);

private:
    int pool_size;
    int stride;
    int calculate_output_dim(int input_dim);
};

#endif

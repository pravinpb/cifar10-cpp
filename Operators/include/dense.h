#ifndef DENSE_H
#define DENSE_H

#include <vector>

class Dense {
public:
    Dense(int input_size, int output_size);
    
    std::vector<float> forward(const std::vector<float>& input);
    void set_weights(const std::vector<std::vector<float>>& weights);
    void set_biases(const std::vector<float>& biases);

private:
    int input_size;
    int output_size;
    std::vector<std::vector<float>> weights;  // Weights matrix
    std::vector<float> biases;               // Bias vector

    float relu(float x);
};

#endif

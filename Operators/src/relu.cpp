#include <algorithm>
#include <cmath>
#include <vector>

void relu(std::vector<float>& tensor) {
    for (auto& value : tensor) {
        value = std::max(0.0f, value);
    }
}
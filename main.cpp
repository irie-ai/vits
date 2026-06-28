#include <torch/torch.h>
#include <iostream>
#include "commons.hpp"

int main() {
    std::vector<int64_t> test = { 1, 2, 3 };

    std::cout << "Original vector: " << test;

    return 0;
}

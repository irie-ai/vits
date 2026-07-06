#pragma once

#include <torch/torch.h>

namespace monotonic_align
{
    torch::Tensor maximumPath(
        const torch::Tensor& negCent,
        const torch::Tensor& mask);
}

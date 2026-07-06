#pragma once

#include <torch/torch.h>

#include <tuple>
#include <vector>

namespace losses
{
    using FeatureMaps = std::vector<std::vector<torch::Tensor>>;

    torch::Tensor featureLoss(
        const FeatureMaps& fmapReal,
        const FeatureMaps& fmapGenerated);

    std::tuple<torch::Tensor, std::vector<double>, std::vector<double>> discriminatorLoss(
        const std::vector<torch::Tensor>& discRealOutputs,
        const std::vector<torch::Tensor>& discGeneratedOutputs);

    std::pair<torch::Tensor, std::vector<torch::Tensor>> generatorLoss(
        const std::vector<torch::Tensor>& discOutputs);

    torch::Tensor klLoss(
        const torch::Tensor& zP,
        const torch::Tensor& logsQ,
        const torch::Tensor& mP,
        const torch::Tensor& logsP,
        const torch::Tensor& zMask);
}

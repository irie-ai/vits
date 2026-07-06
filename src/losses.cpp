#include "losses.hpp"

#include <stdexcept>

namespace losses
{
    torch::Tensor featureLoss(
        const FeatureMaps& fmapReal,
        const FeatureMaps& fmapGenerated)
    {
        if (fmapReal.size() != fmapGenerated.size())
        {
            throw std::invalid_argument("featureLoss expects the same number of discriminator feature map groups.");
        }

        torch::Tensor loss;
        for (size_t i = 0; i < fmapReal.size(); ++i)
        {
            if (fmapReal[i].size() != fmapGenerated[i].size())
            {
                throw std::invalid_argument("featureLoss expects matching feature map counts per discriminator.");
            }

            for (size_t j = 0; j < fmapReal[i].size(); ++j)
            {
                auto real = fmapReal[i][j].to(torch::kFloat32).detach();
                auto generated = fmapGenerated[i][j].to(torch::kFloat32);
                auto itemLoss = torch::mean(torch::abs(real - generated));
                loss = loss.defined() ? loss + itemLoss : itemLoss;
            }
        }

        if (!loss.defined())
        {
            return torch::zeros({});
        }
        return loss * 2.0;
    }

    std::tuple<torch::Tensor, std::vector<double>, std::vector<double>> discriminatorLoss(
        const std::vector<torch::Tensor>& discRealOutputs,
        const std::vector<torch::Tensor>& discGeneratedOutputs)
    {
        if (discRealOutputs.size() != discGeneratedOutputs.size())
        {
            throw std::invalid_argument("discriminatorLoss expects matching discriminator output counts.");
        }

        torch::Tensor loss;
        std::vector<double> realLosses;
        std::vector<double> generatedLosses;
        realLosses.reserve(discRealOutputs.size());
        generatedLosses.reserve(discGeneratedOutputs.size());

        for (size_t i = 0; i < discRealOutputs.size(); ++i)
        {
            auto real = discRealOutputs[i].to(torch::kFloat32);
            auto generated = discGeneratedOutputs[i].to(torch::kFloat32);
            auto realLoss = torch::mean(torch::pow(1.0 - real, 2.0));
            auto generatedLoss = torch::mean(torch::pow(generated, 2.0));
            loss = loss.defined() ? loss + realLoss + generatedLoss : realLoss + generatedLoss;
            realLosses.push_back(realLoss.item<double>());
            generatedLosses.push_back(generatedLoss.item<double>());
        }

        if (!loss.defined())
        {
            loss = torch::zeros({});
        }
        return {loss, realLosses, generatedLosses};
    }

    std::pair<torch::Tensor, std::vector<torch::Tensor>> generatorLoss(
        const std::vector<torch::Tensor>& discOutputs)
    {
        torch::Tensor loss;
        std::vector<torch::Tensor> generatedLosses;
        generatedLosses.reserve(discOutputs.size());

        for (const auto& output : discOutputs)
        {
            auto generated = output.to(torch::kFloat32);
            auto itemLoss = torch::mean(torch::pow(1.0 - generated, 2.0));
            generatedLosses.push_back(itemLoss);
            loss = loss.defined() ? loss + itemLoss : itemLoss;
        }

        if (!loss.defined())
        {
            loss = torch::zeros({});
        }
        return {loss, generatedLosses};
    }

    torch::Tensor klLoss(
        const torch::Tensor& zP,
        const torch::Tensor& logsQ,
        const torch::Tensor& mP,
        const torch::Tensor& logsP,
        const torch::Tensor& zMask)
    {
        auto zPFloat = zP.to(torch::kFloat32);
        auto logsQFloat = logsQ.to(torch::kFloat32);
        auto mPFloat = mP.to(torch::kFloat32);
        auto logsPFloat = logsP.to(torch::kFloat32);
        auto zMaskFloat = zMask.to(torch::kFloat32);

        auto kl = logsPFloat - logsQFloat - 0.5;
        kl = kl + 0.5 * torch::pow(zPFloat - mPFloat, 2.0) * torch::exp(-2.0 * logsPFloat);
        kl = torch::sum(kl * zMaskFloat);
        return kl / torch::sum(zMaskFloat);
    }
}

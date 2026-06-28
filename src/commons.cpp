#include "commons.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <torch/nn/functional/padding.h>

namespace commons
{
    namespace
    {
        torch::Tensor toMaskDtype(const torch::Tensor& tensor)
        {
            return tensor.to(torch::kFloat32);
        }
    }

    void initWeights(torch::nn::Module& module, double mean, double std)
    {
        for (const auto& child : module.modules(/*include_self=*/true))
        {
            const std::string moduleName = child->name();
            if (moduleName.find("Conv") == std::string::npos)
            {
                continue;
            }

            for (auto& namedParameter : child->named_parameters(/*recurse=*/false))
            {
                if (namedParameter.key().find("weight") != std::string::npos)
                {
                    namedParameter.value().data().normal_(mean, std);
                }
            }
        }
    }

    int64_t getPadding(int64_t kernelSize, int64_t dilation)
    {
        return (kernelSize * dilation - dilation) / 2;
    }

    std::vector<int64_t> intersperse(const std::vector<int64_t>& list, int64_t item)
    {
        std::vector<int64_t> result(list.size() * 2 + 1, item);
        for (size_t index = 0; index < list.size(); ++index)
        {
            result[index * 2 + 1] = list[index];
        }
        return result;
    }

    torch::Tensor klDivergence(
        const torch::Tensor& mP,
        const torch::Tensor& logsP,
        const torch::Tensor& mQ,
        const torch::Tensor& logsQ)
    {
        auto kl = (logsQ - logsP) - 0.5;
        kl = kl + 0.5 * (torch::exp(2.0 * logsP) + torch::pow(mP - mQ, 2.0)) * torch::exp(-2.0 * logsQ);
        return kl;
    }

    std::vector<int64_t> convertPadShape(const std::vector<std::vector<int64_t>>& padShape)
    {
        std::vector<int64_t> result;
        result.reserve(padShape.size() * 2);

        for (auto outer = padShape.rbegin(); outer != padShape.rend(); ++outer)
        {
            result.insert(result.end(), outer->begin(), outer->end());
        }

        return result;
    }

    torch::Tensor sliceSegments(const torch::Tensor& x, const torch::Tensor& idsStr, int64_t segmentSize)
    {
        if (x.dim() != 3)
        {
            throw std::invalid_argument("sliceSegments expects x with shape [batch, channels, time].");
        }

        const auto batch = x.size(0);
        auto result = torch::zeros_like(x.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, segmentSize)}));

        for (int64_t batchIndex = 0; batchIndex < batch; ++batchIndex)
        {
            const int64_t start = idsStr.index({batchIndex}).item<int64_t>();
            result.index_put_(
                {batchIndex, torch::indexing::Slice(), torch::indexing::Slice()},
                x.index({batchIndex, torch::indexing::Slice(), torch::indexing::Slice(start, start + segmentSize)}));
        }

        return result;
    }

    std::pair<torch::Tensor, torch::Tensor> randSliceSegments(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& xLengths,
        int64_t segmentSize)
    {
        if (x.dim() != 3)
        {
            throw std::invalid_argument("randSliceSegments expects x with shape [batch, channels, time].");
        }

        const auto batch = x.size(0);
        const auto time = x.size(2);
        auto lengths = xLengths.has_value()
            ? xLengths.value().to(x.device(), torch::kLong)
            : torch::full({batch}, time, torch::TensorOptions().dtype(torch::kLong).device(x.device()));

        auto idsStrMax = lengths - segmentSize + 1;
        auto idsStr = (torch::rand({batch}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())) * idsStrMax.to(torch::kFloat32)).to(torch::kLong);

        return {sliceSegments(x, idsStr, segmentSize), idsStr};
    }

    torch::Tensor randGumbel(torch::IntArrayRef shape)
    {
        auto uniformSamples = torch::rand(shape) * 0.99998 + 0.00001;
        return -torch::log(-torch::log(uniformSamples));
    }

    torch::Tensor randGumbelLike(const torch::Tensor& x)
    {
        return randGumbel(x.sizes()).to(x.options());
    }

    torch::Tensor getTimingSignal1D(int64_t length, int64_t channels, double minTimescale, double maxTimescale)
    {
        const auto numTimescales = channels / 2;
        auto position = torch::arange(length, torch::TensorOptions().dtype(torch::kFloat32));

        if (numTimescales == 0)
        {
            return torch::zeros({1, channels, length}, torch::TensorOptions().dtype(torch::kFloat32));
        }

        const double logTimescaleIncrement = numTimescales > 1
            ? std::log(maxTimescale / minTimescale) / static_cast<double>(numTimescales - 1)
            : 0.0;

        auto invTimescales = minTimescale * torch::exp(
            torch::arange(numTimescales, torch::TensorOptions().dtype(torch::kFloat32)) * -logTimescaleIncrement);
        auto scaledTime = position.unsqueeze(0) * invTimescales.unsqueeze(1);
        auto signal = torch::cat({torch::sin(scaledTime), torch::cos(scaledTime)}, 0);

        if (channels % 2 == 1)
        {
            signal = torch::constant_pad_nd(signal, {0, 0, 0, 1}, 0.0);
        }

        return signal.view({1, channels, length});
    }

    torch::Tensor addTimingSignal1D(const torch::Tensor& x, double minTimescale, double maxTimescale)
    {
        auto signal = getTimingSignal1D(x.size(2), x.size(1), minTimescale, maxTimescale).to(x.options());
        return x + signal;
    }

    torch::Tensor catTimingSignal1D(const torch::Tensor& x, double minTimescale, double maxTimescale, int64_t axis)
    {
        auto signal = getTimingSignal1D(x.size(2), x.size(1), minTimescale, maxTimescale).to(x.options());
        signal = signal.repeat({x.size(0), 1, 1});
        return torch::cat({x, signal}, axis);
    }

    torch::Tensor subsequentMask(int64_t length)
    {
        return torch::tril(torch::ones({length, length}, torch::TensorOptions().dtype(torch::kFloat32))).unsqueeze(0).unsqueeze(0);
    }

    torch::Tensor sequenceMask(const torch::Tensor& length, c10::optional<int64_t> maxLength)
    {
        const int64_t resolvedMaxLength = maxLength.has_value()
            ? maxLength.value()
            : length.max().item<int64_t>();

        auto x = torch::arange(resolvedMaxLength, torch::TensorOptions().dtype(length.scalar_type()).device(length.device()));
        return x.unsqueeze(0) < length.unsqueeze(1);
    }

    torch::Tensor fusedAddTanhSigmoidMultiply(
        const torch::Tensor& inputA,
        const torch::Tensor& inputB,
        int64_t nChannels)
    {
        auto inAct = inputA + inputB;
        auto tAct = torch::tanh(inAct.index({torch::indexing::Slice(), torch::indexing::Slice(0, nChannels), torch::indexing::Slice()}));
        auto sAct = torch::sigmoid(inAct.index({torch::indexing::Slice(), torch::indexing::Slice(nChannels, 2 * nChannels), torch::indexing::Slice()}));
        return tAct * sAct;
    }

    torch::Tensor shift1D(const torch::Tensor& x)
    {
        auto padded = torch::constant_pad_nd(x, convertPadShape({{0, 0}, {0, 0}, {1, 0}}), 0.0);
        return padded.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, -1)});
    }

    torch::Tensor generatePath(const torch::Tensor& duration, const torch::Tensor& mask)
    {
        const auto batch = mask.size(0);
        const auto textLength = mask.size(1);
        const auto frameLength = mask.size(2);

        auto cumDuration = torch::cumsum(duration, 1);
        auto cumDurationFlat = cumDuration.view({batch * textLength});
        auto path = sequenceMask(cumDurationFlat, frameLength).to(mask.scalar_type()).view({batch, textLength, frameLength});
        auto padded = torch::constant_pad_nd(path, convertPadShape({{0, 0}, {1, 0}, {0, 0}}), 0.0);
        path = path - padded.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1), torch::indexing::Slice()});
        return path * mask;
    }

    double clipGradValue(const std::vector<torch::Tensor>& parameters, double clipValue, double normType)
    {
        double totalNorm = 0.0;

        for (const auto& parameter : parameters)
        {
            if (!parameter.defined())
            {
                continue;
            }

            auto grad = parameter.grad();
            if (!grad.defined())
            {
                continue;
            }

            const double paramNorm = grad.data().norm(normType).item<double>();
            totalNorm += std::pow(paramNorm, normType);
            grad.data().clamp_(-clipValue, clipValue);
        }

        return std::pow(totalNorm, 1.0 / normType);
    }
}

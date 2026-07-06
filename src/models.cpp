#include "models.hpp"

#include "commons.hpp"
#include "monotonic_align.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace models
{
    namespace
    {
        constexpr int kFlowAffine = 0;
        constexpr int kFlowConv = 1;
        constexpr int kFlowFlip = 2;
        constexpr double kLogTwoPi = 1.8378770664093453;

        std::pair<torch::Tensor, torch::Tensor> applyFlowStep(
            const StochasticDurationPredictorImpl::FlowStep& step,
            std::vector<modules::ElementwiseAffine>& affineFlows,
            std::vector<modules::ConvFlow>& convFlows,
            std::vector<modules::Flip>& flipFlows,
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g,
            bool reverse)
        {
            if (step.type == kFlowAffine)
            {
                return affineFlows[step.index]->forward(x, xMask, reverse);
            }
            if (step.type == kFlowConv)
            {
                return convFlows[step.index]->forward(x, xMask, g, reverse);
            }
            return flipFlows[step.index]->forward(x, reverse);
        }

    }

    TextEncoderImpl::TextEncoderImpl(
        int64_t nVocab,
        int64_t outChannels,
        int64_t hiddenChannels,
        int64_t filterChannels,
        int64_t nHeads,
        int64_t nLayers,
        int64_t kernelSize,
        double pDropout)
        : nVocab_(nVocab),
          outChannels_(outChannels),
          hiddenChannels_(hiddenChannels),
          filterChannels_(filterChannels),
          nHeads_(nHeads),
          nLayers_(nLayers),
          kernelSize_(kernelSize),
          pDropout_(pDropout)
    {
        emb_ = register_module("emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(nVocab_, hiddenChannels_)));
        encoder_ = register_module(
            "encoder",
            attentions::Encoder(hiddenChannels_, filterChannels_, nHeads_, nLayers_, kernelSize_, pDropout_));
        proj_ = register_module(
            "proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(hiddenChannels_, outChannels_ * 2, 1)));

        torch::NoGradGuard noGrad;
        emb_->weight.normal_(0.0, std::pow(static_cast<double>(hiddenChannels_), -0.5));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TextEncoderImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xLengths)
    {
        auto encoded = emb_->forward(x) * std::sqrt(static_cast<double>(hiddenChannels_));
        encoded = encoded.transpose(1, 2);

        auto xMask = commons::sequenceMask(xLengths, encoded.size(2))
            .unsqueeze(1)
            .to(encoded.options());

        encoded = encoder_->forward(encoded * xMask, xMask);
        auto stats = proj_->forward(encoded) * xMask;

        auto m = stats.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, outChannels_),
            torch::indexing::Slice()});
        auto logs = stats.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(outChannels_, 2 * outChannels_),
            torch::indexing::Slice()});

        return {encoded, m, logs, xMask};
    }

    DurationPredictorImpl::DurationPredictorImpl(
        int64_t inChannels,
        int64_t filterChannels,
        int64_t kernelSize,
        double pDropout,
        int64_t ginChannels)
        : inChannels_(inChannels),
          filterChannels_(filterChannels),
          kernelSize_(kernelSize),
          pDropout_(pDropout),
          ginChannels_(ginChannels)
    {
        const int64_t padding = kernelSize_ / 2;
        drop_ = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(pDropout_)));
        conv1_ = register_module(
            "conv_1",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels_, filterChannels_, kernelSize_).padding(padding)));
        norm1_ = register_module("norm_1", modules::LayerNorm(filterChannels_));
        conv2_ = register_module(
            "conv_2",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(filterChannels_, filterChannels_, kernelSize_).padding(padding)));
        norm2_ = register_module("norm_2", modules::LayerNorm(filterChannels_));
        proj_ = register_module(
            "proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(filterChannels_, 1, 1)));

        if (ginChannels_ != 0)
        {
            cond_ = register_module(
                "cond",
                torch::nn::Conv1d(torch::nn::Conv1dOptions(ginChannels_, inChannels_, 1)));
        }
    }

    torch::Tensor DurationPredictorImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g)
    {
        auto y = x.detach();
        if (g.has_value())
        {
            if (cond_.is_empty())
            {
                throw std::invalid_argument("DurationPredictor received conditioning tensor but ginChannels is zero.");
            }
            y = y + cond_->forward(g.value().detach());
        }

        y = conv1_->forward(y * xMask);
        y = torch::relu(y);
        y = norm1_->forward(y);
        y = drop_->forward(y);
        y = conv2_->forward(y * xMask);
        y = torch::relu(y);
        y = norm2_->forward(y);
        y = drop_->forward(y);
        y = proj_->forward(y * xMask);
        return y * xMask;
    }

    StochasticDurationPredictorImpl::StochasticDurationPredictorImpl(
        int64_t inChannels,
        int64_t filterChannels,
        int64_t kernelSize,
        double pDropout,
        int64_t nFlows,
        int64_t ginChannels)
        : inChannels_(inChannels),
          filterChannels_(filterChannels),
          kernelSize_(kernelSize),
          pDropout_(pDropout),
          nFlows_(nFlows),
          ginChannels_(ginChannels)
    {
        pre_ = register_module(
            "pre",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels_, filterChannels_, 1)));
        convs_ = register_module(
            "convs",
            modules::DDSConv(filterChannels_, kernelSize_, 3, pDropout_));
        proj_ = register_module(
            "proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(filterChannels_, filterChannels_, 1)));

        if (ginChannels_ != 0)
        {
            cond_ = register_module(
                "cond",
                torch::nn::Conv1d(torch::nn::Conv1dOptions(ginChannels_, filterChannels_, 1)));
        }

        postPre_ = register_module(
            "post_pre",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(1, filterChannels_, 1)));
        postConvs_ = register_module(
            "post_convs",
            modules::DDSConv(filterChannels_, kernelSize_, 3, pDropout_));
        postProj_ = register_module(
            "post_proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(filterChannels_, filterChannels_, 1)));
        logFlow_ = register_module("log_flow", modules::Log());

        auto addMainAffine = [&]() {
            auto flow = modules::ElementwiseAffine(2);
            const int index = static_cast<int>(affineFlows_.size());
            affineFlows_.push_back(register_module("flows_affine_" + std::to_string(index), flow));
            flowSteps_.push_back({kFlowAffine, index});
        };
        auto addMainConv = [&]() {
            auto flow = modules::ConvFlow(2, filterChannels_, kernelSize_, 3);
            const int index = static_cast<int>(convFlows_.size());
            convFlows_.push_back(register_module("flows_conv_" + std::to_string(index), flow));
            flowSteps_.push_back({kFlowConv, index});
        };
        auto addMainFlip = [&]() {
            auto flow = modules::Flip();
            const int index = static_cast<int>(flipFlows_.size());
            flipFlows_.push_back(register_module("flows_flip_" + std::to_string(index), flow));
            flowSteps_.push_back({kFlowFlip, index});
        };
        auto addPostAffine = [&]() {
            auto flow = modules::ElementwiseAffine(2);
            const int index = static_cast<int>(postAffineFlows_.size());
            postAffineFlows_.push_back(register_module("post_flows_affine_" + std::to_string(index), flow));
            postFlowSteps_.push_back({kFlowAffine, index});
        };
        auto addPostConv = [&]() {
            auto flow = modules::ConvFlow(2, filterChannels_, kernelSize_, 3);
            const int index = static_cast<int>(postConvFlows_.size());
            postConvFlows_.push_back(register_module("post_flows_conv_" + std::to_string(index), flow));
            postFlowSteps_.push_back({kFlowConv, index});
        };
        auto addPostFlip = [&]() {
            auto flow = modules::Flip();
            const int index = static_cast<int>(postFlipFlows_.size());
            postFlipFlows_.push_back(register_module("post_flows_flip_" + std::to_string(index), flow));
            postFlowSteps_.push_back({kFlowFlip, index});
        };

        addMainAffine();
        addPostAffine();
        for (int64_t i = 0; i < nFlows_; ++i)
        {
            addMainConv();
            addMainFlip();
            addPostConv();
            addPostFlip();
        }
    }

    torch::Tensor StochasticDurationPredictorImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& w,
        const c10::optional<torch::Tensor>& g,
        bool reverse,
        double noiseScale)
    {
        auto h = pre_->forward(x.detach());
        if (g.has_value())
        {
            if (cond_.is_empty())
            {
                throw std::invalid_argument("StochasticDurationPredictor received conditioning tensor but ginChannels is zero.");
            }
            h = h + cond_->forward(g.value().detach());
        }
        h = convs_->forward(h, xMask);
        h = proj_->forward(h) * xMask;

        if (reverse)
        {
            auto z = torch::randn({x.size(0), 2, x.size(2)}, x.options()) * noiseScale;
            z = z * xMask;

            std::vector<FlowStep> reverseSteps(flowSteps_.rbegin(), flowSteps_.rend());
            if (reverseSteps.size() >= 3)
            {
                reverseSteps.erase(reverseSteps.end() - 2);
            }

            for (const auto& step : reverseSteps)
            {
                z = applyFlowStep(step, affineFlows_, convFlows_, flipFlows_, z, xMask, h, true).first;
            }

            return z.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(0, 1),
                torch::indexing::Slice()});
        }

        if (!w.has_value())
        {
            throw std::invalid_argument("StochasticDurationPredictor forward requires w when reverse is false.");
        }

        auto hW = postPre_->forward(w.value().detach());
        hW = postConvs_->forward(hW, xMask, h);
        hW = postProj_->forward(hW) * xMask;

        auto eQ = torch::randn({w.value().size(0), 2, w.value().size(2)}, w.value().options()) * xMask;
        auto zQ = eQ;
        auto logdetTotQ = torch::zeros({x.size(0)}, x.options());
        auto postCondition = h + hW;
        for (const auto& step : postFlowSteps_)
        {
            auto result = applyFlowStep(step, postAffineFlows_, postConvFlows_, postFlipFlows_, zQ, xMask, postCondition, false);
            zQ = result.first;
            logdetTotQ = logdetTotQ + result.second;
        }

        auto zU = zQ.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, 1),
            torch::indexing::Slice()});
        auto z1 = zQ.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(1, 2),
            torch::indexing::Slice()});
        auto u = torch::sigmoid(zU) * xMask;
        auto z0 = (w.value() - u) * xMask;
        logdetTotQ = logdetTotQ +
            (-(torch::softplus(-zU)) - torch::softplus(zU)).mul(xMask).sum({1, 2});
        auto logQ = (-0.5 * (kLogTwoPi + eQ * eQ) * xMask).sum({1, 2}) - logdetTotQ;

        auto logFlowResult = logFlow_->forward(z0, xMask, false);
        z0 = logFlowResult.first;
        auto logdetTot = logFlowResult.second;

        auto z = torch::cat({z0, z1}, 1);
        for (const auto& step : flowSteps_)
        {
            auto result = applyFlowStep(step, affineFlows_, convFlows_, flipFlows_, z, xMask, h, false);
            z = result.first;
            logdetTot = logdetTot + result.second;
        }

        auto nll = (0.5 * (kLogTwoPi + z * z) * xMask).sum({1, 2}) - logdetTot;
        return nll + logQ;
    }

    PosteriorEncoderImpl::PosteriorEncoderImpl(
        int64_t inChannels,
        int64_t outChannels,
        int64_t hiddenChannels,
        int64_t kernelSize,
        int64_t dilationRate,
        int64_t nLayers,
        int64_t ginChannels)
        : inChannels_(inChannels),
          outChannels_(outChannels),
          hiddenChannels_(hiddenChannels),
          kernelSize_(kernelSize),
          dilationRate_(dilationRate),
          nLayers_(nLayers),
          ginChannels_(ginChannels)
    {
        pre_ = register_module(
            "pre",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels_, hiddenChannels_, 1)));
        enc_ = register_module(
            "enc",
            modules::WN(hiddenChannels_, kernelSize_, dilationRate_, nLayers_, ginChannels_));
        proj_ = register_module(
            "proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(hiddenChannels_, outChannels_ * 2, 1)));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PosteriorEncoderImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xLengths,
        const c10::optional<torch::Tensor>& g)
    {
        auto xMask = commons::sequenceMask(xLengths, x.size(2))
            .unsqueeze(1)
            .to(x.options());

        auto y = pre_->forward(x) * xMask;
        y = enc_->forward(y, xMask, g);
        auto stats = proj_->forward(y) * xMask;

        auto m = stats.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, outChannels_),
            torch::indexing::Slice()});
        auto logs = stats.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(outChannels_, 2 * outChannels_),
            torch::indexing::Slice()});
        auto z = (m + torch::randn_like(m) * torch::exp(logs)) * xMask;

        return {z, m, logs, xMask};
    }

    ResidualCouplingBlockImpl::ResidualCouplingBlockImpl(
        int64_t channels,
        int64_t hiddenChannels,
        int64_t kernelSize,
        int64_t dilationRate,
        int64_t nLayers,
        int64_t nFlows,
        int64_t ginChannels)
        : channels_(channels),
          hiddenChannels_(hiddenChannels),
          kernelSize_(kernelSize),
          dilationRate_(dilationRate),
          nLayers_(nLayers),
          nFlows_(nFlows),
          ginChannels_(ginChannels)
    {
        couplingLayers_.reserve(nFlows_);
        flipLayers_.reserve(nFlows_);

        for (int64_t i = 0; i < nFlows_; ++i)
        {
            auto coupling = modules::ResidualCouplingLayer(
                channels_,
                hiddenChannels_,
                kernelSize_,
                dilationRate_,
                nLayers_,
                0.0,
                ginChannels_,
                true);
            auto flip = modules::Flip();

            couplingLayers_.push_back(register_module("flow_" + std::to_string(i) + "_coupling", coupling));
            flipLayers_.push_back(register_module("flow_" + std::to_string(i) + "_flip", flip));
        }
    }

    torch::Tensor ResidualCouplingBlockImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g,
        bool reverse)
    {
        auto y = x;

        if (!reverse)
        {
            for (int64_t i = 0; i < nFlows_; ++i)
            {
                y = couplingLayers_[i]->forward(y, xMask, g, false).first;
                y = flipLayers_[i]->forward(y, false).first;
            }
            return y;
        }

        for (int64_t i = nFlows_ - 1; i >= 0; --i)
        {
            y = flipLayers_[i]->forward(y, true).first;
            y = couplingLayers_[i]->forward(y, xMask, g, true).first;
        }
        return y;
    }

    GeneratorImpl::GeneratorImpl(
        int64_t initialChannel,
        const std::string& resblock,
        const std::vector<int64_t>& resblockKernelSizes,
        const std::vector<std::vector<int64_t>>& resblockDilationSizes,
        const std::vector<int64_t>& upsampleRates,
        int64_t upsampleInitialChannel,
        const std::vector<int64_t>& upsampleKernelSizes,
        int64_t ginChannels)
        : initialChannel_(initialChannel),
          resblock_(resblock),
          resblockKernelSizes_(resblockKernelSizes),
          resblockDilationSizes_(resblockDilationSizes),
          upsampleRates_(upsampleRates),
          upsampleInitialChannel_(upsampleInitialChannel),
          upsampleKernelSizes_(upsampleKernelSizes),
          ginChannels_(ginChannels),
          numKernels_(static_cast<int64_t>(resblockKernelSizes_.size())),
          numUpsamples_(static_cast<int64_t>(upsampleRates_.size()))
    {
        if (resblock_ != "1" && resblock_ != "2")
        {
            throw std::invalid_argument("Generator resblock must be \"1\" or \"2\".");
        }
        if (resblockKernelSizes_.size() != resblockDilationSizes_.size())
        {
            throw std::invalid_argument("Generator resblock kernel and dilation lists must have the same size.");
        }
        if (upsampleRates_.size() != upsampleKernelSizes_.size())
        {
            throw std::invalid_argument("Generator upsample rate and kernel lists must have the same size.");
        }

        convPre_ = register_module(
            "conv_pre",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(initialChannel_, upsampleInitialChannel_, 7).padding(3)));

        ups_.reserve(numUpsamples_);
        resblocks1_.reserve(numUpsamples_ * numKernels_);
        resblocks2_.reserve(numUpsamples_ * numKernels_);

        for (int64_t i = 0; i < numUpsamples_; ++i)
        {
            const int64_t inChannels = upsampleInitialChannel_ / (1LL << i);
            const int64_t outChannels = upsampleInitialChannel_ / (1LL << (i + 1));
            const int64_t kernelSize = upsampleKernelSizes_[i];
            const int64_t stride = upsampleRates_[i];
            auto up = torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(inChannels, outChannels, kernelSize)
                                                     .stride(stride)
                                                     .padding((kernelSize - stride) / 2));
            // TODO: Python applies weight_norm and init_weights to upsamplers.
            // Weight norm is not ported yet, so this keeps forward behavior first.
            ups_.push_back(register_module("ups_" + std::to_string(i), up));

            for (int64_t j = 0; j < numKernels_; ++j)
            {
                const int64_t index = i * numKernels_ + j;
                if (resblock_ == "1")
                {
                    auto block = modules::ResBlock1(outChannels, resblockKernelSizes_[j], resblockDilationSizes_[j]);
                    resblocks1_.push_back(register_module("resblocks_" + std::to_string(index), block));
                }
                else
                {
                    auto block = modules::ResBlock2(outChannels, resblockKernelSizes_[j], resblockDilationSizes_[j]);
                    resblocks2_.push_back(register_module("resblocks_" + std::to_string(index), block));
                }
            }
        }

        const int64_t postChannels = upsampleInitialChannel_ / (1LL << numUpsamples_);
        convPost_ = register_module(
            "conv_post",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(postChannels, 1, 7).padding(3).bias(false)));

        if (ginChannels_ != 0)
        {
            cond_ = register_module(
                "cond",
                torch::nn::Conv1d(torch::nn::Conv1dOptions(ginChannels_, upsampleInitialChannel_, 1)));
        }
    }

    torch::Tensor GeneratorImpl::forward(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& g)
    {
        auto y = convPre_->forward(x);
        if (g.has_value())
        {
            if (cond_.is_empty())
            {
                throw std::invalid_argument("Generator received conditioning tensor but ginChannels is zero.");
            }
            y = y + cond_->forward(g.value());
        }

        for (int64_t i = 0; i < numUpsamples_; ++i)
        {
            y = torch::leaky_relu(y, 0.1);
            y = ups_[i]->forward(y);

            torch::Tensor ys;
            for (int64_t j = 0; j < numKernels_; ++j)
            {
                const int64_t index = i * numKernels_ + j;
                auto blockOut = resblock_ == "1"
                    ? resblocks1_[index]->forward(y)
                    : resblocks2_[index]->forward(y);
                ys = ys.defined() ? ys + blockOut : blockOut;
            }
            y = ys / static_cast<double>(numKernels_);
        }

        y = torch::leaky_relu(y, 0.1);
        y = convPost_->forward(y);
        return torch::tanh(y);
    }

    void GeneratorImpl::removeWeightNorm()
    {
        for (auto& block : resblocks1_)
        {
            block->removeWeightNorm();
        }
        for (auto& block : resblocks2_)
        {
            block->removeWeightNorm();
        }
    }

    DiscriminatorPImpl::DiscriminatorPImpl(
        int64_t period,
        int64_t kernelSize,
        int64_t stride,
        bool useSpectralNorm)
        : period_(period),
          kernelSize_(kernelSize),
          stride_(stride),
          useSpectralNorm_(useSpectralNorm)
    {
        // TODO: Python wraps these convs with weight_norm or spectral_norm.
        // Norm wrappers are not ported yet, so keep the discriminator topology first.
        const int64_t pad = commons::getPadding(kernelSize_, 1);
        const std::vector<std::pair<int64_t, int64_t>> channels = {
            {1, 32},
            {32, 128},
            {128, 512},
            {512, 1024},
            {1024, 1024},
        };

        convs_.reserve(channels.size());
        for (size_t i = 0; i < channels.size(); ++i)
        {
            const bool last = i + 1 == channels.size();
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels[i].first, channels[i].second, {kernelSize_, 1})
                    .stride(last ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{stride_, 1})
                    .padding({pad, 0}));
            convs_.push_back(register_module("convs_" + std::to_string(i), conv));
        }

        convPost_ = register_module(
            "conv_post",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1, {3, 1}).padding({1, 0})));
    }

    DiscriminatorOutput DiscriminatorPImpl::forward(const torch::Tensor& x)
    {
        std::vector<torch::Tensor> featureMaps;
        auto y = x;
        const int64_t batch = y.size(0);
        const int64_t channels = y.size(1);
        int64_t time = y.size(2);

        if (time % period_ != 0)
        {
            const int64_t pad = period_ - (time % period_);
            y = torch::constant_pad_nd(y, {0, pad}, 0.0);
            time += pad;
        }

        y = y.contiguous().view({batch, channels, time / period_, period_});
        for (auto& conv : convs_)
        {
            y = conv->forward(y);
            y = torch::leaky_relu(y, 0.1);
            featureMaps.push_back(y);
        }

        y = convPost_->forward(y);
        featureMaps.push_back(y);
        return {y.flatten(1, -1), featureMaps};
    }

    DiscriminatorSImpl::DiscriminatorSImpl(bool useSpectralNorm)
        : useSpectralNorm_(useSpectralNorm)
    {
        // TODO: Python wraps these convs with weight_norm or spectral_norm.
        // Norm wrappers are not ported yet, so keep the discriminator topology first.
        struct ConvSpec
        {
            int64_t inChannels;
            int64_t outChannels;
            int64_t kernelSize;
            int64_t stride;
            int64_t groups;
            int64_t padding;
        };
        const std::vector<ConvSpec> specs = {
            {1, 16, 15, 1, 1, 7},
            {16, 64, 41, 4, 4, 20},
            {64, 256, 41, 4, 16, 20},
            {256, 1024, 41, 4, 64, 20},
            {1024, 1024, 41, 4, 256, 20},
            {1024, 1024, 5, 1, 1, 2},
        };

        convs_.reserve(specs.size());
        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto& spec = specs[i];
            auto conv = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(spec.inChannels, spec.outChannels, spec.kernelSize)
                    .stride(spec.stride)
                    .groups(spec.groups)
                    .padding(spec.padding));
            convs_.push_back(register_module("convs_" + std::to_string(i), conv));
        }

        convPost_ = register_module(
            "conv_post",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(1024, 1, 3).padding(1)));
    }

    DiscriminatorOutput DiscriminatorSImpl::forward(const torch::Tensor& x)
    {
        std::vector<torch::Tensor> featureMaps;
        auto y = x;

        for (auto& conv : convs_)
        {
            y = conv->forward(y);
            y = torch::leaky_relu(y, 0.1);
            featureMaps.push_back(y);
        }

        y = convPost_->forward(y);
        featureMaps.push_back(y);
        return {y.flatten(1, -1), featureMaps};
    }

    MultiPeriodDiscriminatorImpl::MultiPeriodDiscriminatorImpl(bool useSpectralNorm)
        : useSpectralNorm_(useSpectralNorm)
    {
        discS_ = register_module("discriminator_s", DiscriminatorS(useSpectralNorm_));

        const std::vector<int64_t> periods = {2, 3, 5, 7, 11};
        discPs_.reserve(periods.size());
        for (size_t i = 0; i < periods.size(); ++i)
        {
            auto disc = DiscriminatorP(periods[i], 5, 3, useSpectralNorm_);
            discPs_.push_back(register_module("discriminator_p_" + std::to_string(i), disc));
        }
    }

    MultiPeriodDiscriminatorOutput MultiPeriodDiscriminatorImpl::forward(
        const torch::Tensor& y,
        const torch::Tensor& yHat)
    {
        std::vector<torch::Tensor> yDRs;
        std::vector<torch::Tensor> yDGs;
        std::vector<std::vector<torch::Tensor>> fmapRs;
        std::vector<std::vector<torch::Tensor>> fmapGs;

        auto realScale = discS_->forward(y);
        auto generatedScale = discS_->forward(yHat);
        yDRs.push_back(realScale.first);
        yDGs.push_back(generatedScale.first);
        fmapRs.push_back(realScale.second);
        fmapGs.push_back(generatedScale.second);

        for (auto& disc : discPs_)
        {
            auto realPeriod = disc->forward(y);
            auto generatedPeriod = disc->forward(yHat);
            yDRs.push_back(realPeriod.first);
            yDGs.push_back(generatedPeriod.first);
            fmapRs.push_back(realPeriod.second);
            fmapGs.push_back(generatedPeriod.second);
        }

        return {yDRs, yDGs, fmapRs, fmapGs};
    }

    SynthesizerTrnImpl::SynthesizerTrnImpl(
        int64_t nVocab,
        int64_t specChannels,
        int64_t segmentSize,
        int64_t interChannels,
        int64_t hiddenChannels,
        int64_t filterChannels,
        int64_t nHeads,
        int64_t nLayers,
        int64_t kernelSize,
        double pDropout,
        const std::string& resblock,
        const std::vector<int64_t>& resblockKernelSizes,
        const std::vector<std::vector<int64_t>>& resblockDilationSizes,
        const std::vector<int64_t>& upsampleRates,
        int64_t upsampleInitialChannel,
        const std::vector<int64_t>& upsampleKernelSizes,
        int64_t nSpeakers,
        int64_t ginChannels,
        bool useSdp)
        : nVocab_(nVocab),
          specChannels_(specChannels),
          segmentSize_(segmentSize),
          interChannels_(interChannels),
          hiddenChannels_(hiddenChannels),
          filterChannels_(filterChannels),
          nHeads_(nHeads),
          nLayers_(nLayers),
          kernelSize_(kernelSize),
          pDropout_(pDropout),
          nSpeakers_(nSpeakers),
          ginChannels_(ginChannels),
          useSdp_(useSdp)
    {
        encP_ = register_module(
            "enc_p",
            TextEncoder(nVocab_, interChannels_, hiddenChannels_, filterChannels_, nHeads_, nLayers_, kernelSize_, pDropout_));
        encQ_ = register_module(
            "enc_q",
            PosteriorEncoder(specChannels_, interChannels_, hiddenChannels_, 5, 1, 16, ginChannels_));
        dec_ = register_module(
            "dec",
            Generator(interChannels_, resblock, resblockKernelSizes, resblockDilationSizes, upsampleRates, upsampleInitialChannel, upsampleKernelSizes, ginChannels_));
        flow_ = register_module(
            "flow",
            ResidualCouplingBlock(interChannels_, hiddenChannels_, 5, 1, 4, 4, ginChannels_));

        if (useSdp_)
        {
            dpSdp_ = register_module(
                "dp",
                StochasticDurationPredictor(hiddenChannels_, 192, 3, 0.5, 4, ginChannels_));
        }
        else
        {
            dp_ = register_module(
                "dp",
                DurationPredictor(hiddenChannels_, 256, 3, 0.5, ginChannels_));
        }

        if (nSpeakers_ > 0)
        {
            embG_ = register_module("emb_g", torch::nn::Embedding(torch::nn::EmbeddingOptions(nSpeakers_, ginChannels_)));
        }
    }

    SynthesizerForwardResult SynthesizerTrnImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xLengths,
        const torch::Tensor& y,
        const torch::Tensor& yLengths,
        const c10::optional<torch::Tensor>& sid)
    {
        auto encResult = encP_->forward(x, xLengths);
        auto hText = std::get<0>(encResult);
        auto mPText = std::get<1>(encResult);
        auto logsPText = std::get<2>(encResult);
        auto xMask = std::get<3>(encResult);

        torch::Tensor g;
        if (sid.has_value())
        {
            if (embG_.is_empty())
            {
                throw std::invalid_argument("SynthesizerTrn received sid but nSpeakers is zero.");
            }
            g = embG_->forward(sid.value()).unsqueeze(-1);
        }

        auto posteriorResult = encQ_->forward(y, yLengths, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt);
        auto z = std::get<0>(posteriorResult);
        auto mQ = std::get<1>(posteriorResult);
        auto logsQ = std::get<2>(posteriorResult);
        auto yMask = std::get<3>(posteriorResult);

        auto zP = flow_->forward(z, yMask, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt, false);

        auto sPSqR = torch::exp(-2.0 * logsPText);
        auto negCent1 = (-0.5 * kLogTwoPi - logsPText).sum({1}, true);
        auto negCent2 = torch::matmul((-0.5 * zP.pow(2)).transpose(1, 2), sPSqR);
        auto negCent3 = torch::matmul(zP.transpose(1, 2), mPText * sPSqR);
        auto negCent4 = (-0.5 * mPText.pow(2) * sPSqR).sum({1}, true);
        auto negCent = negCent1 + negCent2 + negCent3 + negCent4;
        auto attentionMask = xMask.unsqueeze(2) * yMask.unsqueeze(-1);
        auto attention = monotonic_align::maximumPath(negCent, attentionMask.squeeze(1)).unsqueeze(1).detach();
        auto w = attention.sum(2);

        torch::Tensor lengthLoss;
        if (useSdp_)
        {
            lengthLoss = dpSdp_->forward(hText, xMask, w, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt, false);
            lengthLoss = lengthLoss / torch::sum(xMask);
        }
        else
        {
            auto logwTarget = torch::log(w + 1.0e-6) * xMask;
            auto logw = dp_->forward(hText, xMask, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt);
            lengthLoss = (logw - logwTarget).pow(2).sum({1, 2}) / torch::sum(xMask);
        }

        auto attentionSqueezed = attention.squeeze(1);
        auto mP = torch::matmul(attentionSqueezed, mPText.transpose(1, 2)).transpose(1, 2);
        auto logsP = torch::matmul(attentionSqueezed, logsPText.transpose(1, 2)).transpose(1, 2);

        auto sliceResult = commons::randSliceSegments(z, yLengths, segmentSize_);
        auto zSlice = sliceResult.first;
        auto idsSlice = sliceResult.second;
        auto audio = dec_->forward(zSlice, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt);

        return {audio, lengthLoss, attention, idsSlice, xMask, yMask, z, zP, mP, logsP, mQ, logsQ};
    }

    SynthesizerInferResult SynthesizerTrnImpl::infer(
        const torch::Tensor& x,
        const torch::Tensor& xLengths,
        const c10::optional<torch::Tensor>& sid,
        double noiseScale,
        double lengthScale,
        double noiseScaleW,
        c10::optional<int64_t> maxLength)
    {
        torch::Tensor g;
        if (sid.has_value())
        {
            if (embG_.is_empty())
            {
                throw std::invalid_argument("SynthesizerTrn received sid but nSpeakers <= 1.");
            }
            g = embG_->forward(sid.value()).unsqueeze(-1);
        }

        auto encResult = encP_->forward(x, xLengths);
        auto hText = std::get<0>(encResult);
        auto mPText = std::get<1>(encResult);
        auto logsPText = std::get<2>(encResult);
        auto xMask = std::get<3>(encResult);

        torch::Tensor logw;
        if (useSdp_)
        {
            logw = dpSdp_->forward(hText, xMask, c10::nullopt, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt, true, noiseScaleW);
        }
        else
        {
            logw = dp_->forward(hText, xMask, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt);
        }

        auto w = torch::exp(logw) * xMask * lengthScale;
        auto wCeil = torch::ceil(w);
        auto yLengths = torch::clamp_min(wCeil.sum({1, 2}), 1).to(torch::kLong);
        auto yMask = commons::sequenceMask(yLengths, c10::nullopt).unsqueeze(1).to(xMask.options());
        auto pathMask = xMask.transpose(1, 2) * yMask;
        auto attention = commons::generatePath(wCeil.squeeze(1), pathMask);

        auto attentionBt = attention.transpose(1, 2);
        auto mP = torch::matmul(attentionBt, mPText.transpose(1, 2)).transpose(1, 2);
        auto logsP = torch::matmul(attentionBt, logsPText.transpose(1, 2)).transpose(1, 2);

        auto zP = (mP + torch::randn_like(mP) * torch::exp(logsP) * noiseScale) * yMask;
        auto z = flow_->forward(zP, yMask, g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt, true);

        if (maxLength.has_value() && z.size(2) > maxLength.value())
        {
            z = z.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(0, maxLength.value())});
            yMask = yMask.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(0, maxLength.value())});
        }

        auto audio = dec_->forward((z * yMask), g.defined() ? c10::optional<torch::Tensor>(g) : c10::nullopt);
        return {audio, attention.unsqueeze(1), yMask, z, zP, mP, logsP};
    }
}

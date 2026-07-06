#include "modules.hpp"

#include "commons.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace modules
{
    namespace
    {
        torch::nn::Conv1d makeConv1d(
            int64_t inChannels,
            int64_t outChannels,
            int64_t kernelSize,
            int64_t dilation = 1,
            int64_t groups = 1)
        {
            torch::nn::Conv1d conv(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize)
                .padding(commons::getPadding(kernelSize, dilation))
                .dilation(dilation)
                .groups(groups));
            // TODO: Python applies init_weights to Conv modules. commons::initWeights(*conv)
            // crashed with this Windows/LibTorch setup, so it is temporarily skipped.
            // Restore safe initialization before supporting C++ training.
            return conv;
        }

        torch::Tensor sumChannelsAndTime(const torch::Tensor& x)
        {
            return x.sum({1, 2});
        }

        torch::Tensor gatherLastDim(const torch::Tensor& values, const torch::Tensor& indices)
        {
            return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1);
        }

        std::tuple<torch::Tensor, torch::Tensor> rationalQuadraticSpline(
            const torch::Tensor& inputs,
            const torch::Tensor& unnormalizedWidths,
            const torch::Tensor& unnormalizedHeights,
            const torch::Tensor& unnormalizedDerivatives,
            bool inverse,
            double tailBound)
        {
            constexpr double minBinWidth = 1.0e-3;
            constexpr double minBinHeight = 1.0e-3;
            constexpr double minDerivative = 1.0e-3;

            const int64_t numBins = unnormalizedWidths.size(-1);
            const double left = -tailBound;
            const double right = tailBound;
            const double bottom = -tailBound;
            const double top = tailBound;
            const double widthRange = right - left;
            const double heightRange = top - bottom;

            auto inside = (inputs >= left) & (inputs <= right);

            auto widths = torch::softmax(unnormalizedWidths, -1);
            widths = minBinWidth + (widthRange - minBinWidth * numBins) * widths;
            auto cumwidths = torch::cumsum(widths, -1);
            cumwidths = torch::constant_pad_nd(cumwidths, {1, 0}, 0.0) + left;
            cumwidths.index_put_({torch::indexing::Ellipsis, 0}, left);
            cumwidths.index_put_({torch::indexing::Ellipsis, numBins}, right);
            widths = cumwidths.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None)}) -
                cumwidths.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, -1)});

            auto heights = torch::softmax(unnormalizedHeights, -1);
            heights = minBinHeight + (heightRange - minBinHeight * numBins) * heights;
            auto cumheights = torch::cumsum(heights, -1);
            cumheights = torch::constant_pad_nd(cumheights, {1, 0}, 0.0) + bottom;
            cumheights.index_put_({torch::indexing::Ellipsis, 0}, bottom);
            cumheights.index_put_({torch::indexing::Ellipsis, numBins}, top);
            heights = cumheights.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None)}) -
                cumheights.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, -1)});

            auto derivatives = minDerivative + torch::softplus(unnormalizedDerivatives);
            derivatives = torch::constant_pad_nd(derivatives, {1, 1}, 1.0);
            derivatives.index_put_({torch::indexing::Ellipsis, 0}, 1.0);
            derivatives.index_put_({torch::indexing::Ellipsis, numBins}, 1.0);

            auto binLookupValues = inverse ? cumheights : cumwidths;
            auto binIdx = (inputs.unsqueeze(-1) >= binLookupValues.index({
                torch::indexing::Ellipsis,
                torch::indexing::Slice(1, torch::indexing::None)})).sum(-1).to(torch::kLong);
            binIdx = torch::clamp(binIdx, 0, numBins - 1);

            auto inputCumwidths = gatherLastDim(cumwidths, binIdx);
            auto inputBinWidths = gatherLastDim(widths, binIdx);
            auto inputCumheights = gatherLastDim(cumheights, binIdx);
            auto inputHeights = gatherLastDim(heights, binIdx);
            auto inputDelta = inputHeights / inputBinWidths;
            auto inputDerivatives = gatherLastDim(derivatives, binIdx);
            auto inputDerivativesPlusOne = gatherLastDim(derivatives, binIdx + 1);

            auto forwardFromTheta = [&](const torch::Tensor& theta) {
                auto thetaOneMinus = theta * (1.0 - theta);
                auto numerator = inputHeights *
                    (inputDelta * theta * theta + inputDerivatives * thetaOneMinus);
                auto denominator = inputDelta +
                    (inputDerivatives + inputDerivativesPlusOne - 2.0 * inputDelta) * thetaOneMinus;
                auto outputs = inputCumheights + numerator / denominator;
                auto derivativeNumerator = inputDelta * inputDelta *
                    (inputDerivativesPlusOne * theta * theta +
                        2.0 * inputDelta * thetaOneMinus +
                        inputDerivatives * (1.0 - theta) * (1.0 - theta));
                auto logabsdet = torch::log(derivativeNumerator) - 2.0 * torch::log(denominator);
                return std::make_pair(outputs, logabsdet);
            };

            torch::Tensor outputs;
            torch::Tensor logabsdet;
            if (!inverse)
            {
                auto theta = (inputs - inputCumwidths) / inputBinWidths;
                auto transformed = forwardFromTheta(theta);
                outputs = transformed.first;
                logabsdet = transformed.second;
            }
            else
            {
                auto lower = torch::zeros_like(inputs);
                auto upper = torch::ones_like(inputs);
                for (int64_t i = 0; i < 32; ++i)
                {
                    auto mid = (lower + upper) * 0.5;
                    auto midOutputs = forwardFromTheta(mid).first;
                    lower = torch::where(midOutputs <= inputs, mid, lower);
                    upper = torch::where(midOutputs > inputs, mid, upper);
                }
                auto theta = (lower + upper) * 0.5;
                auto transformed = forwardFromTheta(theta);
                outputs = inputCumwidths + theta * inputBinWidths;
                logabsdet = -transformed.second;
            }

            outputs = torch::where(inside, outputs, inputs);
            logabsdet = torch::where(inside, logabsdet, torch::zeros_like(logabsdet));
            return {outputs, logabsdet};
        }
    }

    LayerNormImpl::LayerNormImpl(int64_t channels, double eps)
        : channels_(channels), eps_(eps)
    {
        weight_ = register_parameter("gamma", torch::ones({channels}));
        bias_ = register_parameter("beta", torch::zeros({channels}));
    }

    torch::Tensor LayerNormImpl::forward(const torch::Tensor& x)
    {
        auto y = x.transpose(1, -1);
        y = torch::layer_norm(y, {channels_}, weight_, bias_, eps_);
        return y.transpose(1, -1);
    }

    ConvReluNormImpl::ConvReluNormImpl(
        int64_t inChannels,
        int64_t hiddenChannels,
        int64_t outChannels,
        int64_t kernelSize,
        int64_t nLayers,
        double pDropout)
    {
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));
        // TODO: Submodules should normally be registered with register_module.
        // Manual registration segfaulted in this environment, so ModuleHolders
        // are kept in vectors for now. Revisit before weight save/load support.

        for (int64_t i = 0; i < nLayers; ++i)
        {
            const int64_t inLayerChannels = i == 0 ? inChannels : hiddenChannels;
            auto conv = makeConv1d(inLayerChannels, hiddenChannels, kernelSize);
            auto norm = LayerNorm(hiddenChannels);
            convLayers_.push_back(conv);
            normLayers_.push_back(norm);
        }

        proj_ = torch::nn::Conv1d(torch::nn::Conv1dOptions(hiddenChannels, outChannels, 1));
        // TODO: Python zero-initializes proj.weight and proj.bias.
        // data().zero_() crashed at runtime, so restore this with a safe no_grad path.
    }

    torch::Tensor ConvReluNormImpl::forward(const torch::Tensor& x, const torch::Tensor& xMask)
    {
        auto y = x;
        const auto xOrg = x;

        for (size_t i = 0; i < convLayers_.size(); ++i)
        {
            y = convLayers_[i]->forward(y * xMask);
            y = normLayers_[i]->forward(y);
            y = torch::relu(y);
            y = dropout_->forward(y);
        }

        y = proj_->forward(y);
        return xOrg + y * xMask;
    }

    DDSConvImpl::DDSConvImpl(int64_t channels, int64_t kernelSize, int64_t nLayers, double pDropout)
    {
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));
        // TODO: conv/norm layers should be registered modules. Manual registration
        // was unstable, so this keeps forward behavior testable first.

        for (int64_t i = 0; i < nLayers; ++i)
        {
            const int64_t dilation = static_cast<int64_t>(std::pow(kernelSize, i));
            auto convSep = makeConv1d(channels, channels, kernelSize, dilation, channels);
            auto conv1x1 = makeConv1d(channels, channels, 1);
            auto norm1 = LayerNorm(channels);
            auto norm2 = LayerNorm(channels);
            convSepLayers_.push_back(convSep);
            conv1x1Layers_.push_back(conv1x1);
            normLayers1_.push_back(norm1);
            normLayers2_.push_back(norm2);
        }
    }

    torch::Tensor DDSConvImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g)
    {
        auto y = x;
        if (g.has_value())
        {
            y = y + g.value();
        }

        for (size_t i = 0; i < convSepLayers_.size(); ++i)
        {
            auto z = convSepLayers_[i]->forward(y * xMask);
            z = normLayers1_[i]->forward(z);
            z = torch::gelu(z);
            z = conv1x1Layers_[i]->forward(z);
            z = normLayers2_[i]->forward(z);
            z = torch::gelu(z);
            z = dropout_->forward(z);
            y = y + z;
        }

        return y * xMask;
    }

    WNImpl::WNImpl(
        int64_t hiddenChannels,
        int64_t kernelSize,
        int64_t dilationRate,
        int64_t nLayers,
        int64_t ginChannels,
        double pDropout)
        : hiddenChannels_(hiddenChannels), nLayers_(nLayers), ginChannels_(ginChannels)
    {
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));
        // TODO: in_layers/res_skip_layers/cond_layer should be registered modules.
        // Restore registration when adding state_dict and weight_norm support.

        if (ginChannels_ != 0)
        {
            condLayer_ = torch::nn::Conv1d(
                torch::nn::Conv1dOptions(ginChannels_, 2 * hiddenChannels_ * nLayers_, 1));
        }

        for (int64_t i = 0; i < nLayers_; ++i)
        {
            const int64_t dilation = static_cast<int64_t>(std::pow(dilationRate, i));
            auto inLayer = makeConv1d(hiddenChannels_, 2 * hiddenChannels_, kernelSize, dilation);
            const int64_t resSkipChannels = i < nLayers_ - 1 ? 2 * hiddenChannels_ : hiddenChannels_;
            auto resSkipLayer = makeConv1d(hiddenChannels_, resSkipChannels, 1);
            inLayers_.push_back(inLayer);
            resSkipLayers_.push_back(resSkipLayer);
        }
    }

    torch::Tensor WNImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g,
        bool)
    {
        auto y = x;
        auto output = torch::zeros_like(y);
        torch::Tensor gLayer;

        if (g.has_value())
        {
            if (condLayer_.is_empty())
            {
                throw std::invalid_argument("WN received conditioning tensor but ginChannels is zero.");
            }
            gLayer = condLayer_->forward(g.value());
        }

        for (int64_t i = 0; i < nLayers_; ++i)
        {
            auto xIn = inLayers_[i]->forward(y);

            if (gLayer.defined())
            {
                const int64_t offset = i * 2 * hiddenChannels_;
                xIn = xIn + gLayer.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(offset, offset + 2 * hiddenChannels_),
                    torch::indexing::Slice()});
            }

            auto acts = commons::fusedAddTanhSigmoidMultiply(
                xIn,
                torch::zeros_like(xIn),
                hiddenChannels_);
            acts = dropout_->forward(acts);

            auto resSkipActs = resSkipLayers_[i]->forward(acts);
            if (i < nLayers_ - 1)
            {
                auto resActs = resSkipActs.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(0, hiddenChannels_),
                    torch::indexing::Slice()});
                auto skipActs = resSkipActs.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(hiddenChannels_, 2 * hiddenChannels_),
                    torch::indexing::Slice()});
                y = (y + resActs) * xMask;
                output = output + skipActs;
            }
            else
            {
                output = output + resSkipActs;
            }
        }

        return output * xMask;
    }

    void WNImpl::removeWeightNorm()
    {
    }

    ResBlock1Impl::ResBlock1Impl(int64_t channels, int64_t kernelSize, const std::vector<int64_t>& dilation)
        : lreluSlope_(0.1)
    {
        // TODO: Python ResBlock1 registers these convs with weight_norm.
        // Current code skips weight_norm and manual registration; handle this
        // before loading inference weights directly.
        for (const auto d : dilation)
        {
            const auto index = convs1_.size();
            auto conv1 = makeConv1d(channels, channels, kernelSize, d);
            auto conv2 = makeConv1d(channels, channels, kernelSize, 1);
            convs1_.push_back(conv1);
            convs2_.push_back(conv2);
        }
    }

    torch::Tensor ResBlock1Impl::forward(const torch::Tensor& x, const c10::optional<torch::Tensor>& xMask)
    {
        auto y = x;
        for (size_t i = 0; i < convs1_.size(); ++i)
        {
            auto xt = torch::leaky_relu(y, lreluSlope_);
            if (xMask.has_value())
            {
                xt = xt * xMask.value();
            }
            xt = convs1_[i]->forward(xt);
            xt = torch::leaky_relu(xt, lreluSlope_);
            if (xMask.has_value())
            {
                xt = xt * xMask.value();
            }
            xt = convs2_[i]->forward(xt);
            y = y + xt;
            if (xMask.has_value())
            {
                y = y * xMask.value();
            }
        }
        return y;
    }

    void ResBlock1Impl::removeWeightNorm()
    {
    }

    ResBlock2Impl::ResBlock2Impl(int64_t channels, int64_t kernelSize, const std::vector<int64_t>& dilation)
        : lreluSlope_(0.1)
    {
        // TODO: Python ResBlock2 registers these convs with weight_norm.
        // Current code skips weight_norm and manual registration.
        for (const auto d : dilation)
        {
            const auto index = convs_.size();
            auto conv = makeConv1d(channels, channels, kernelSize, d);
            convs_.push_back(conv);
        }
    }

    torch::Tensor ResBlock2Impl::forward(const torch::Tensor& x, const c10::optional<torch::Tensor>& xMask)
    {
        auto y = x;
        for (size_t i = 0; i < convs_.size(); ++i)
        {
            auto xt = torch::leaky_relu(y, lreluSlope_);
            if (xMask.has_value())
            {
                xt = xt * xMask.value();
            }
            xt = convs_[i]->forward(xt);
            y = y + xt;
            if (xMask.has_value())
            {
                y = y * xMask.value();
            }
        }
        return y;
    }

    void ResBlock2Impl::removeWeightNorm()
    {
    }

    std::pair<torch::Tensor, torch::Tensor> LogImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        bool reverse)
    {
        if (!reverse)
        {
            auto y = torch::log(torch::clamp_min(x, 1.0e-5)) * xMask;
            return {y, sumChannelsAndTime(-y)};
        }

        return {torch::exp(x) * xMask, torch::zeros({x.size(0)}, x.options())};
    }

    std::pair<torch::Tensor, torch::Tensor> FlipImpl::forward(const torch::Tensor& x, bool)
    {
        return {torch::flip(x, {1}), torch::zeros({x.size(0)}, x.options())};
    }

    ElementwiseAffineImpl::ElementwiseAffineImpl(int64_t channels)
    {
        m_ = register_parameter("m", torch::zeros({channels, 1}));
        logs_ = register_parameter("logs", torch::zeros({channels, 1}));
    }

    std::pair<torch::Tensor, torch::Tensor> ElementwiseAffineImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        bool reverse)
    {
        if (!reverse)
        {
            auto y = m_ + torch::exp(logs_) * x;
            y = y * xMask;
            return {y, sumChannelsAndTime(logs_ * xMask)};
        }

        auto y = (x - m_) * torch::exp(-logs_) * xMask;
        return {y, torch::zeros({x.size(0)}, x.options())};
    }

    ResidualCouplingLayerImpl::ResidualCouplingLayerImpl(
        int64_t channels,
        int64_t hiddenChannels,
        int64_t kernelSize,
        int64_t dilationRate,
        int64_t nLayers,
        double pDropout,
        int64_t ginChannels,
        bool meanOnly)
        : halfChannels_(channels / 2), meanOnly_(meanOnly)
    {
        if (channels % 2 != 0)
        {
            throw std::invalid_argument("ResidualCouplingLayer channels must be even.");
        }

        pre_ = torch::nn::Conv1d(torch::nn::Conv1dOptions(halfChannels_, hiddenChannels, 1));
        enc_ = WN(hiddenChannels, kernelSize, dilationRate, nLayers, ginChannels, pDropout);
        post_ = torch::nn::Conv1d(torch::nn::Conv1dOptions(
            hiddenChannels,
            halfChannels_ * (meanOnly_ ? 1 : 2),
            1));
        // TODO: Python zero-initializes post.weight/post.bias and registers
        // pre/enc/post. zero_ and register_module were unstable here, so forward
        // stability is prioritized for now.
    }

    std::pair<torch::Tensor, torch::Tensor> ResidualCouplingLayerImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g,
        bool reverse)
    {
        auto x0 = x.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, halfChannels_),
            torch::indexing::Slice()});
        auto x1 = x.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(halfChannels_, 2 * halfChannels_),
            torch::indexing::Slice()});

        auto h = pre_->forward(x0) * xMask;
        h = enc_->forward(h, xMask, g);
        auto stats = post_->forward(h) * xMask;

        torch::Tensor m;
        torch::Tensor logs;
        if (meanOnly_)
        {
            m = stats;
            logs = torch::zeros_like(m);
        }
        else
        {
            m = stats.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(0, halfChannels_),
                torch::indexing::Slice()});
            logs = stats.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(halfChannels_, 2 * halfChannels_),
                torch::indexing::Slice()});
        }

        if (!reverse)
        {
            x1 = m + x1 * torch::exp(logs) * xMask;
            return {torch::cat({x0, x1}, 1), sumChannelsAndTime(logs)};
        }

        x1 = (x1 - m) * torch::exp(-logs) * xMask;
        return {torch::cat({x0, x1}, 1), torch::zeros({x.size(0)}, x.options())};
    }

    ConvFlowImpl::ConvFlowImpl(
        int64_t inChannels,
        int64_t filterChannels,
        int64_t kernelSize,
        int64_t nLayers,
        int64_t numBins,
        double tailBound)
        : inChannels_(inChannels),
          filterChannels_(filterChannels),
          kernelSize_(kernelSize),
          nLayers_(nLayers),
          numBins_(numBins),
          tailBound_(tailBound),
          halfChannels_(inChannels / 2)
    {
        if (inChannels_ % 2 != 0)
        {
            throw std::invalid_argument("ConvFlow inChannels must be even.");
        }

        pre_ = register_module(
            "pre",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(halfChannels_, filterChannels_, 1)));
        convs_ = register_module(
            "convs",
            DDSConv(filterChannels_, kernelSize_, nLayers_, 0.0));
        proj_ = register_module(
            "proj",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(
                filterChannels_,
                halfChannels_ * (numBins_ * 3 - 1),
                1)));
        // TODO: Python zero-initializes proj.weight/proj.bias. Keep default
        // initialization until no_grad zeroing is verified across configurations.
    }

    std::pair<torch::Tensor, torch::Tensor> ConvFlowImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const c10::optional<torch::Tensor>& g,
        bool reverse)
    {
        auto x0 = x.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(0, halfChannels_),
            torch::indexing::Slice()});
        auto x1 = x.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(halfChannels_, inChannels_),
            torch::indexing::Slice()});

        auto h = pre_->forward(x0);
        h = convs_->forward(h, xMask, g);
        h = proj_->forward(h) * xMask;

        const int64_t batch = x0.size(0);
        const int64_t channels = x0.size(1);
        const int64_t time = x0.size(2);
        h = h.view({batch, channels, numBins_ * 3 - 1, time}).permute({0, 1, 3, 2});

        auto unnormalizedWidths = h.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(0, numBins_)}) / std::sqrt(static_cast<double>(filterChannels_));
        auto unnormalizedHeights = h.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(numBins_, 2 * numBins_)}) / std::sqrt(static_cast<double>(filterChannels_));
        auto unnormalizedDerivatives = h.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(2 * numBins_, torch::indexing::None)});

        auto transformed = rationalQuadraticSpline(
            x1,
            unnormalizedWidths,
            unnormalizedHeights,
            unnormalizedDerivatives,
            reverse,
            tailBound_);
        x1 = std::get<0>(transformed);
        auto logdet = sumChannelsAndTime(std::get<1>(transformed) * xMask);

        auto y = torch::cat({x0, x1}, 1) * xMask;
        if (reverse)
        {
            logdet = torch::zeros({x.size(0)}, x.options());
        }
        return {y, logdet};
    }
}

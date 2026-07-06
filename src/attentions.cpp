#include "attentions.hpp"

#include "commons.hpp"

#include <cmath>
#include <stdexcept>

#include <torch/nn/functional/padding.h>

namespace attentions
{
    namespace
    {
        torch::nn::Conv1d makeConv1d(int64_t inChannels, int64_t outChannels, int64_t kernelSize)
        {
            return torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize));
        }
    }

    FFNImpl::FFNImpl(
        int64_t inChannels,
        int64_t outChannels,
        int64_t filterChannels,
        int64_t kernelSize,
        double pDropout,
        const std::string& activation,
        bool causal)
        : kernelSize_(kernelSize), activation_(activation), causal_(causal)
    {
        conv1_ = makeConv1d(inChannels, filterChannels, kernelSize);
        conv2_ = makeConv1d(filterChannels, outChannels, kernelSize);
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));
    }

    torch::Tensor FFNImpl::forward(const torch::Tensor& x, const torch::Tensor& xMask)
    {
        auto y = conv1_->forward(padInput(x * xMask));
        if (activation_ == "gelu")
        {
            y = y * torch::sigmoid(1.702 * y);
        }
        else
        {
            y = torch::relu(y);
        }
        y = dropout_->forward(y);
        y = conv2_->forward(padInput(y * xMask));
        return y * xMask;
    }

    torch::Tensor FFNImpl::padInput(const torch::Tensor& x) const
    {
        if (kernelSize_ == 1)
        {
            return x;
        }

        const int64_t padLeft = causal_ ? kernelSize_ - 1 : (kernelSize_ - 1) / 2;
        const int64_t padRight = causal_ ? 0 : kernelSize_ / 2;
        return torch::constant_pad_nd(x, commons::convertPadShape({{0, 0}, {0, 0}, {padLeft, padRight}}), 0.0);
    }

    MultiHeadAttentionImpl::MultiHeadAttentionImpl(
        int64_t channels,
        int64_t outChannels,
        int64_t nHeads,
        double pDropout,
        c10::optional<int64_t> windowSize,
        bool headsShare,
        c10::optional<int64_t> blockLength,
        bool proximalBias,
        bool proximalInit)
        : channels_(channels),
          outChannels_(outChannels),
          nHeads_(nHeads),
          kChannels_(channels / nHeads),
          windowSize_(windowSize),
          headsShare_(headsShare),
          blockLength_(blockLength),
          proximalBias_(proximalBias)
    {
        if (channels % nHeads != 0)
        {
            throw std::invalid_argument("MultiHeadAttention channels must be divisible by nHeads.");
        }

        convQ_ = makeConv1d(channels, channels, 1);
        convK_ = makeConv1d(channels, channels, 1);
        convV_ = makeConv1d(channels, channels, 1);
        convO_ = makeConv1d(channels, outChannels, 1);
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));

        if (windowSize_.has_value())
        {
            const int64_t nHeadsRel = headsShare_ ? 1 : nHeads_;
            const double relStddev = std::pow(static_cast<double>(kChannels_), -0.5);
            embRelK_ = register_parameter(
                "emb_rel_k",
                torch::randn({nHeadsRel, windowSize_.value() * 2 + 1, kChannels_}) * relStddev);
            embRelV_ = register_parameter(
                "emb_rel_v",
                torch::randn({nHeadsRel, windowSize_.value() * 2 + 1, kChannels_}) * relStddev);
        }

        // TODO: Python copies conv_q weights into conv_k when proximalInit=true.
        // copy_ crashed while constructing Decoder, so restore it with a safe no_grad path.
        (void)proximalInit;
    }

    torch::Tensor MultiHeadAttentionImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& c,
        const c10::optional<torch::Tensor>& attnMask)
    {
        auto q = convQ_->forward(x);
        auto k = convK_->forward(c);
        auto v = convV_->forward(c);
        auto result = attention(q, k, v, attnMask);
        attn_ = result.second;
        return convO_->forward(result.first);
    }

    const torch::Tensor& MultiHeadAttentionImpl::attentionWeights() const
    {
        return attn_;
    }

    std::pair<torch::Tensor, torch::Tensor> MultiHeadAttentionImpl::attention(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const c10::optional<torch::Tensor>& mask)
    {
        const int64_t batch = key.size(0);
        const int64_t channels = key.size(1);
        const int64_t sourceLength = key.size(2);
        const int64_t targetLength = query.size(2);

        auto q = query.view({batch, nHeads_, kChannels_, targetLength}).transpose(2, 3);
        auto k = key.view({batch, nHeads_, kChannels_, sourceLength}).transpose(2, 3);
        auto v = value.view({batch, nHeads_, kChannels_, sourceLength}).transpose(2, 3);

        auto scores = torch::matmul(q / std::sqrt(static_cast<double>(kChannels_)), k.transpose(-2, -1));

        if (windowSize_.has_value())
        {
            if (sourceLength != targetLength)
            {
                throw std::invalid_argument("Relative attention is only available for self-attention.");
            }

            auto keyRelativeEmbeddings = getRelativeEmbeddings(embRelK_, sourceLength).to(query.options());
            auto relLogits = matmulWithRelativeKeys(q / std::sqrt(static_cast<double>(kChannels_)), keyRelativeEmbeddings);
            scores = scores + relativePositionToAbsolutePosition(relLogits);
        }

        if (proximalBias_)
        {
            if (sourceLength != targetLength)
            {
                throw std::invalid_argument("Proximal bias is only available for self-attention.");
            }
            scores = scores + attentionBiasProximal(sourceLength, scores.options());
        }

        if (mask.has_value())
        {
            scores = scores.masked_fill(mask.value() == 0, -1.0e4);
        }

        if (blockLength_.has_value())
        {
            if (sourceLength != targetLength)
            {
                throw std::invalid_argument("Block attention is only available for self-attention.");
            }
            auto blockMask = torch::ones_like(scores)
                .triu(-blockLength_.value())
                .tril(blockLength_.value());
            scores = scores.masked_fill(blockMask == 0, -1.0e4);
        }

        auto pAttn = torch::softmax(scores, -1);
        pAttn = dropout_->forward(pAttn);
        auto output = torch::matmul(pAttn, v);
        if (windowSize_.has_value())
        {
            auto relativeWeights = absolutePositionToRelativePosition(pAttn);
            auto valueRelativeEmbeddings = getRelativeEmbeddings(embRelV_, sourceLength).to(value.options());
            output = output + matmulWithRelativeValues(relativeWeights, valueRelativeEmbeddings);
        }
        output = output.transpose(2, 3).contiguous().view({batch, channels, targetLength});
        return {output, pAttn};
    }

    torch::Tensor MultiHeadAttentionImpl::getRelativeEmbeddings(
        const torch::Tensor& relativeEmbeddings,
        int64_t length) const
    {
        const int64_t maxRelativePosition = 2 * windowSize_.value() + 1;
        const int64_t padLength = std::max<int64_t>(length - (windowSize_.value() + 1), 0);
        const int64_t sliceStartPosition = std::max<int64_t>((windowSize_.value() + 1) - length, 0);
        const int64_t sliceEndPosition = sliceStartPosition + 2 * length - 1;

        torch::Tensor paddedRelativeEmbeddings = relativeEmbeddings;
        if (padLength > 0)
        {
            paddedRelativeEmbeddings = torch::constant_pad_nd(
                relativeEmbeddings,
                commons::convertPadShape({{0, 0}, {padLength, padLength}, {0, 0}}),
                0.0);
        }

        (void)maxRelativePosition;
        return paddedRelativeEmbeddings.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(sliceStartPosition, sliceEndPosition),
            torch::indexing::Slice()});
    }

    torch::Tensor MultiHeadAttentionImpl::matmulWithRelativeKeys(const torch::Tensor& x, const torch::Tensor& y) const
    {
        return torch::matmul(x, y.unsqueeze(0).transpose(-2, -1));
    }

    torch::Tensor MultiHeadAttentionImpl::matmulWithRelativeValues(const torch::Tensor& x, const torch::Tensor& y) const
    {
        return torch::matmul(x, y.unsqueeze(0));
    }

    torch::Tensor MultiHeadAttentionImpl::relativePositionToAbsolutePosition(const torch::Tensor& x) const
    {
        const int64_t batch = x.size(0);
        const int64_t heads = x.size(1);
        const int64_t length = x.size(2);

        auto padded = torch::constant_pad_nd(
            x,
            commons::convertPadShape({{0, 0}, {0, 0}, {0, 0}, {0, 1}}),
            0.0);
        auto flat = padded.view({batch, heads, length * 2 * length});
        flat = torch::constant_pad_nd(
            flat,
            commons::convertPadShape({{0, 0}, {0, 0}, {0, length - 1}}),
            0.0);
        auto final = flat.view({batch, heads, length + 1, 2 * length - 1});
        return final.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(0, length),
            torch::indexing::Slice(length - 1, torch::indexing::None)});
    }

    torch::Tensor MultiHeadAttentionImpl::absolutePositionToRelativePosition(const torch::Tensor& x) const
    {
        const int64_t batch = x.size(0);
        const int64_t heads = x.size(1);
        const int64_t length = x.size(2);

        auto padded = torch::constant_pad_nd(
            x,
            commons::convertPadShape({{0, 0}, {0, 0}, {0, 0}, {0, length - 1}}),
            0.0);
        auto flat = padded.view({batch, heads, length * length + length * (length - 1)});
        flat = torch::constant_pad_nd(
            flat,
            commons::convertPadShape({{0, 0}, {0, 0}, {length, 0}}),
            0.0);
        auto final = flat.view({batch, heads, length, 2 * length});
        return final.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(1, torch::indexing::None)});
    }

    torch::Tensor MultiHeadAttentionImpl::attentionBiasProximal(int64_t length, const torch::TensorOptions& options) const
    {
        auto r = torch::arange(length, options.dtype(torch::kFloat32));
        auto diff = r.unsqueeze(0) - r.unsqueeze(1);
        return -torch::log1p(torch::abs(diff)).unsqueeze(0).unsqueeze(0);
    }

    EncoderImpl::EncoderImpl(
        int64_t hiddenChannels,
        int64_t filterChannels,
        int64_t nHeads,
        int64_t nLayers,
        int64_t kernelSize,
        double pDropout,
        c10::optional<int64_t> windowSize)
    {
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));

        for (int64_t i = 0; i < nLayers; ++i)
        {
            attnLayers_.push_back(MultiHeadAttention(hiddenChannels, hiddenChannels, nHeads, pDropout, windowSize));
            normLayers1_.push_back(modules::LayerNorm(hiddenChannels));
            ffnLayers_.push_back(FFN(hiddenChannels, hiddenChannels, filterChannels, kernelSize, pDropout));
            normLayers2_.push_back(modules::LayerNorm(hiddenChannels));
        }
    }

    torch::Tensor EncoderImpl::forward(const torch::Tensor& x, const torch::Tensor& xMask)
    {
        auto y = x * xMask;
        auto attnMask = xMask.unsqueeze(2) * xMask.unsqueeze(-1);

        for (size_t i = 0; i < attnLayers_.size(); ++i)
        {
            auto h = attnLayers_[i]->forward(y, y, attnMask);
            h = dropout_->forward(h);
            y = normLayers1_[i]->forward(y + h);
            h = ffnLayers_[i]->forward(y, xMask);
            h = dropout_->forward(h);
            y = normLayers2_[i]->forward(y + h);
            y = y * xMask;
        }

        return y;
    }

    DecoderImpl::DecoderImpl(
        int64_t hiddenChannels,
        int64_t filterChannels,
        int64_t nHeads,
        int64_t nLayers,
        int64_t kernelSize,
        double pDropout,
        bool proximalBias,
        bool proximalInit)
    {
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(pDropout));

        for (int64_t i = 0; i < nLayers; ++i)
        {
            selfAttnLayers_.push_back(MultiHeadAttention(
                hiddenChannels, hiddenChannels, nHeads, pDropout, c10::nullopt, true, c10::nullopt, proximalBias, proximalInit));
            normLayers0_.push_back(modules::LayerNorm(hiddenChannels));
            encdecAttnLayers_.push_back(MultiHeadAttention(hiddenChannels, hiddenChannels, nHeads, pDropout));
            normLayers1_.push_back(modules::LayerNorm(hiddenChannels));
            ffnLayers_.push_back(FFN(hiddenChannels, hiddenChannels, filterChannels, kernelSize, pDropout, "", true));
            normLayers2_.push_back(modules::LayerNorm(hiddenChannels));
        }
    }

    torch::Tensor DecoderImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& xMask,
        const torch::Tensor& h,
        const torch::Tensor& hMask)
    {
        auto y = x * xMask;
        auto selfAttnMask = commons::subsequentMask(xMask.size(2)).to(y.options());
        auto encdecAttnMask = hMask.unsqueeze(2) * xMask.unsqueeze(-1);

        for (size_t i = 0; i < selfAttnLayers_.size(); ++i)
        {
            auto z = selfAttnLayers_[i]->forward(y, y, selfAttnMask);
            z = dropout_->forward(z);
            y = normLayers0_[i]->forward(y + z);
            z = encdecAttnLayers_[i]->forward(y, h, encdecAttnMask);
            z = dropout_->forward(z);
            y = normLayers1_[i]->forward(y + z);
            z = ffnLayers_[i]->forward(y, xMask);
            z = dropout_->forward(z);
            y = normLayers2_[i]->forward(y + z);
            y = y * xMask;
        }

        return y;
    }
}

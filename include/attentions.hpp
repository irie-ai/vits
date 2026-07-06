#pragma once

#include <torch/torch.h>

#include "modules.hpp"

#include <cstdint>
#include <vector>

namespace attentions
{
    class FFNImpl : public torch::nn::Module
    {
    public:
        FFNImpl(
            int64_t inChannels,
            int64_t outChannels,
            int64_t filterChannels,
            int64_t kernelSize,
            double pDropout = 0.0,
            const std::string& activation = "",
            bool causal = false);

        torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& xMask);

    private:
        torch::Tensor padInput(const torch::Tensor& x) const;

        int64_t kernelSize_;
        std::string activation_;
        bool causal_;
        torch::nn::Conv1d conv1_{nullptr};
        torch::nn::Conv1d conv2_{nullptr};
        torch::nn::Dropout dropout_{nullptr};
    };
    TORCH_MODULE(FFN);

    class MultiHeadAttentionImpl : public torch::nn::Module
    {
    public:
        MultiHeadAttentionImpl(
            int64_t channels,
            int64_t outChannels,
            int64_t nHeads,
            double pDropout = 0.0,
            c10::optional<int64_t> windowSize = c10::nullopt,
            bool headsShare = true,
            c10::optional<int64_t> blockLength = c10::nullopt,
            bool proximalBias = false,
            bool proximalInit = false);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& c,
            const c10::optional<torch::Tensor>& attnMask = c10::nullopt);

        const torch::Tensor& attentionWeights() const;

    private:
        std::pair<torch::Tensor, torch::Tensor> attention(
            const torch::Tensor& query,
            const torch::Tensor& key,
            const torch::Tensor& value,
            const c10::optional<torch::Tensor>& mask);

        torch::Tensor getRelativeEmbeddings(const torch::Tensor& relativeEmbeddings, int64_t length) const;
        torch::Tensor matmulWithRelativeKeys(const torch::Tensor& x, const torch::Tensor& y) const;
        torch::Tensor matmulWithRelativeValues(const torch::Tensor& x, const torch::Tensor& y) const;
        torch::Tensor relativePositionToAbsolutePosition(const torch::Tensor& x) const;
        torch::Tensor absolutePositionToRelativePosition(const torch::Tensor& x) const;
        torch::Tensor attentionBiasProximal(int64_t length, const torch::TensorOptions& options) const;

        int64_t channels_;
        int64_t outChannels_;
        int64_t nHeads_;
        int64_t kChannels_;
        c10::optional<int64_t> windowSize_;
        bool headsShare_;
        c10::optional<int64_t> blockLength_;
        bool proximalBias_;
        torch::nn::Conv1d convQ_{nullptr};
        torch::nn::Conv1d convK_{nullptr};
        torch::nn::Conv1d convV_{nullptr};
        torch::nn::Conv1d convO_{nullptr};
        torch::nn::Dropout dropout_{nullptr};
        torch::Tensor embRelK_;
        torch::Tensor embRelV_;
        torch::Tensor attn_;
    };
    TORCH_MODULE(MultiHeadAttention);

    class EncoderImpl : public torch::nn::Module
    {
    public:
        EncoderImpl(
            int64_t hiddenChannels,
            int64_t filterChannels,
            int64_t nHeads,
            int64_t nLayers,
            int64_t kernelSize = 1,
            double pDropout = 0.0,
            c10::optional<int64_t> windowSize = 4);

        torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& xMask);

    private:
        std::vector<MultiHeadAttention> attnLayers_;
        std::vector<modules::LayerNorm> normLayers1_;
        std::vector<FFN> ffnLayers_;
        std::vector<modules::LayerNorm> normLayers2_;
        torch::nn::Dropout dropout_{nullptr};
    };
    TORCH_MODULE(Encoder);

    class DecoderImpl : public torch::nn::Module
    {
    public:
        DecoderImpl(
            int64_t hiddenChannels,
            int64_t filterChannels,
            int64_t nHeads,
            int64_t nLayers,
            int64_t kernelSize = 1,
            double pDropout = 0.0,
            bool proximalBias = false,
            bool proximalInit = true);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const torch::Tensor& h,
            const torch::Tensor& hMask);

    private:
        std::vector<MultiHeadAttention> selfAttnLayers_;
        std::vector<modules::LayerNorm> normLayers0_;
        std::vector<MultiHeadAttention> encdecAttnLayers_;
        std::vector<modules::LayerNorm> normLayers1_;
        std::vector<FFN> ffnLayers_;
        std::vector<modules::LayerNorm> normLayers2_;
        torch::nn::Dropout dropout_{nullptr};
    };
    TORCH_MODULE(Decoder);
}

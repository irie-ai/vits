#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace modules
{
    class LayerNormImpl : public torch::nn::Module
    {
    public:
        LayerNormImpl(int64_t channels, double eps = 1.0e-5);
        torch::Tensor forward(const torch::Tensor& x);

    private:
        int64_t channels_;
        double eps_;
        torch::Tensor weight_;
        torch::Tensor bias_;
    };
    TORCH_MODULE(LayerNorm);

    class ConvReluNormImpl : public torch::nn::Module
    {
    public:
        ConvReluNormImpl(
            int64_t inChannels,
            int64_t hiddenChannels,
            int64_t outChannels,
            int64_t kernelSize,
            int64_t nLayers,
            double pDropout);

        torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& xMask);

    private:
        std::vector<torch::nn::Conv1d> convLayers_;
        std::vector<LayerNorm> normLayers_;
        torch::nn::Dropout dropout_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
    };
    TORCH_MODULE(ConvReluNorm);

    class DDSConvImpl : public torch::nn::Module
    {
    public:
        DDSConvImpl(
            int64_t channels,
            int64_t kernelSize,
            int64_t nLayers,
            double pDropout = 0.0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt);

    private:
        std::vector<torch::nn::Conv1d> convSepLayers_;
        std::vector<torch::nn::Conv1d> conv1x1Layers_;
        std::vector<LayerNorm> normLayers1_;
        std::vector<LayerNorm> normLayers2_;
        torch::nn::Dropout dropout_{nullptr};
    };
    TORCH_MODULE(DDSConv);

    class WNImpl : public torch::nn::Module
    {
    public:
        WNImpl(
            int64_t hiddenChannels,
            int64_t kernelSize,
            int64_t dilationRate,
            int64_t nLayers,
            int64_t ginChannels = 0,
            double pDropout = 0.0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt,
            bool reverse = false);

        void removeWeightNorm();

    private:
        int64_t hiddenChannels_;
        int64_t nLayers_;
        int64_t ginChannels_;
        std::vector<torch::nn::Conv1d> inLayers_;
        std::vector<torch::nn::Conv1d> resSkipLayers_;
        torch::nn::Conv1d condLayer_{nullptr};
        torch::nn::Dropout dropout_{nullptr};
    };
    TORCH_MODULE(WN);

    class ResBlock1Impl : public torch::nn::Module
    {
    public:
        ResBlock1Impl(
            int64_t channels,
            int64_t kernelSize = 3,
            const std::vector<int64_t>& dilation = {1, 3, 5});

        torch::Tensor forward(
            const torch::Tensor& x,
            const c10::optional<torch::Tensor>& xMask = c10::nullopt);

        void removeWeightNorm();

    private:
        double lreluSlope_;
        std::vector<torch::nn::Conv1d> convs1_;
        std::vector<torch::nn::Conv1d> convs2_;
    };
    TORCH_MODULE(ResBlock1);

    class ResBlock2Impl : public torch::nn::Module
    {
    public:
        ResBlock2Impl(
            int64_t channels,
            int64_t kernelSize = 3,
            const std::vector<int64_t>& dilation = {1, 3});

        torch::Tensor forward(
            const torch::Tensor& x,
            const c10::optional<torch::Tensor>& xMask = c10::nullopt);

        void removeWeightNorm();

    private:
        double lreluSlope_;
        std::vector<torch::nn::Conv1d> convs_;
    };
    TORCH_MODULE(ResBlock2);

    class LogImpl : public torch::nn::Module
    {
    public:
        std::pair<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            bool reverse = false);
    };
    TORCH_MODULE(Log);

    class FlipImpl : public torch::nn::Module
    {
    public:
        std::pair<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            bool reverse = false);
    };
    TORCH_MODULE(Flip);

    class ElementwiseAffineImpl : public torch::nn::Module
    {
    public:
        explicit ElementwiseAffineImpl(int64_t channels);

        std::pair<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            bool reverse = false);

    private:
        torch::Tensor m_;
        torch::Tensor logs_;
    };
    TORCH_MODULE(ElementwiseAffine);

    class ResidualCouplingLayerImpl : public torch::nn::Module
    {
    public:
        ResidualCouplingLayerImpl(
            int64_t channels,
            int64_t hiddenChannels,
            int64_t kernelSize,
            int64_t dilationRate,
            int64_t nLayers,
            double pDropout = 0.0,
            int64_t ginChannels = 0,
            bool meanOnly = false);

        std::pair<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt,
            bool reverse = false);

    private:
        int64_t halfChannels_;
        bool meanOnly_;
        torch::nn::Conv1d pre_{nullptr};
        WN enc_{nullptr};
        torch::nn::Conv1d post_{nullptr};
    };
    TORCH_MODULE(ResidualCouplingLayer);

    class ConvFlowImpl : public torch::nn::Module
    {
    public:
        ConvFlowImpl(
            int64_t inChannels,
            int64_t filterChannels,
            int64_t kernelSize,
            int64_t nLayers,
            int64_t numBins = 10,
            double tailBound = 5.0);

        std::pair<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt,
            bool reverse = false);

    private:
        int64_t inChannels_;
        int64_t filterChannels_;
        int64_t kernelSize_;
        int64_t nLayers_;
        int64_t numBins_;
        double tailBound_;
        int64_t halfChannels_;
        torch::nn::Conv1d pre_{nullptr};
        DDSConv convs_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
    };
    TORCH_MODULE(ConvFlow);
}

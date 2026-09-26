#pragma once

#include <torch/torch.h>

#include "attentions.hpp"
#include "modules.hpp"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace models
{
    class TextEncoderImpl : public torch::nn::Module
    {
    public:
        TextEncoderImpl(
            int64_t nVocab,
            int64_t outChannels,
            int64_t hiddenChannels,
            int64_t filterChannels,
            int64_t nHeads,
            int64_t nLayers,
            int64_t kernelSize,
            double pDropout);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xLengths);

    private:
        int64_t nVocab_;
        int64_t outChannels_;
        int64_t hiddenChannels_;
        int64_t filterChannels_;
        int64_t nHeads_;
        int64_t nLayers_;
        int64_t kernelSize_;
        double pDropout_;
        torch::nn::Embedding emb_{nullptr};
        attentions::Encoder encoder_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
    };
    TORCH_MODULE(TextEncoder);

    class DurationPredictorImpl : public torch::nn::Module
    {
    public:
        DurationPredictorImpl(
            int64_t inChannels,
            int64_t filterChannels,
            int64_t kernelSize,
            double pDropout,
            int64_t ginChannels = 0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt);

    private:
        int64_t inChannels_;
        int64_t filterChannels_;
        int64_t kernelSize_;
        double pDropout_;
        int64_t ginChannels_;
        torch::nn::Dropout drop_{nullptr};
        torch::nn::Conv1d conv1_{nullptr};
        modules::LayerNorm norm1_{nullptr};
        torch::nn::Conv1d conv2_{nullptr};
        modules::LayerNorm norm2_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
        torch::nn::Conv1d cond_{nullptr};
    };
    TORCH_MODULE(DurationPredictor);

    class StochasticDurationPredictorImpl : public torch::nn::Module
    {
    public:
        struct FlowStep
        {
            int type;
            int index;
        };

        StochasticDurationPredictorImpl(
            int64_t inChannels,
            int64_t filterChannels,
            int64_t kernelSize,
            double pDropout,
            int64_t nFlows = 4,
            int64_t ginChannels = 0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& w = c10::nullopt,
            const c10::optional<torch::Tensor>& g = c10::nullopt,
            bool reverse = false,
            double noiseScale = 1.0);

    private:
        int64_t inChannels_;
        int64_t filterChannels_;
        int64_t kernelSize_;
        double pDropout_;
        int64_t nFlows_;
        int64_t ginChannels_;
        torch::nn::Conv1d pre_{nullptr};
        modules::DDSConv convs_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
        torch::nn::Conv1d cond_{nullptr};
        torch::nn::Conv1d postPre_{nullptr};
        modules::DDSConv postConvs_{nullptr};
        torch::nn::Conv1d postProj_{nullptr};
        modules::Log logFlow_{nullptr};
        std::vector<modules::ElementwiseAffine> affineFlows_;
        std::vector<modules::ConvFlow> convFlows_;
        std::vector<modules::Flip> flipFlows_;
        std::vector<FlowStep> flowSteps_;
        std::vector<modules::ElementwiseAffine> postAffineFlows_;
        std::vector<modules::ConvFlow> postConvFlows_;
        std::vector<modules::Flip> postFlipFlows_;
        std::vector<FlowStep> postFlowSteps_;
    };
    TORCH_MODULE(StochasticDurationPredictor);

    class PosteriorEncoderImpl : public torch::nn::Module
    {
    public:
        PosteriorEncoderImpl(
            int64_t inChannels,
            int64_t outChannels,
            int64_t hiddenChannels,
            int64_t kernelSize,
            int64_t dilationRate,
            int64_t nLayers,
            int64_t ginChannels = 0);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& x,
            const torch::Tensor& xLengths,
            const c10::optional<torch::Tensor>& g = c10::nullopt);

    private:
        int64_t inChannels_;
        int64_t outChannels_;
        int64_t hiddenChannels_;
        int64_t kernelSize_;
        int64_t dilationRate_;
        int64_t nLayers_;
        int64_t ginChannels_;
        torch::nn::Conv1d pre_{nullptr};
        modules::WN enc_{nullptr};
        torch::nn::Conv1d proj_{nullptr};
    };
    TORCH_MODULE(PosteriorEncoder);

    class ResidualCouplingBlockImpl : public torch::nn::Module
    {
    public:
        ResidualCouplingBlockImpl(
            int64_t channels,
            int64_t hiddenChannels,
            int64_t kernelSize,
            int64_t dilationRate,
            int64_t nLayers,
            int64_t nFlows = 4,
            int64_t ginChannels = 0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& xMask,
            const c10::optional<torch::Tensor>& g = c10::nullopt,
            bool reverse = false);

    private:
        int64_t channels_;
        int64_t hiddenChannels_;
        int64_t kernelSize_;
        int64_t dilationRate_;
        int64_t nLayers_;
        int64_t nFlows_;
        int64_t ginChannels_;
        std::vector<modules::ResidualCouplingLayer> couplingLayers_;
        std::vector<modules::Flip> flipLayers_;
    };
    TORCH_MODULE(ResidualCouplingBlock);

    class GeneratorImpl : public torch::nn::Module
    {
    public:
        GeneratorImpl(
            int64_t initialChannel,
            const std::string& resblock,
            const std::vector<int64_t>& resblockKernelSizes,
            const std::vector<std::vector<int64_t>>& resblockDilationSizes,
            const std::vector<int64_t>& upsampleRates,
            int64_t upsampleInitialChannel,
            const std::vector<int64_t>& upsampleKernelSizes,
            int64_t ginChannels = 0);

        torch::Tensor forward(
            const torch::Tensor& x,
            const c10::optional<torch::Tensor>& g = c10::nullopt);

        void removeWeightNorm();

    private:
        int64_t initialChannel_;
        std::string resblock_;
        std::vector<int64_t> resblockKernelSizes_;
        std::vector<std::vector<int64_t>> resblockDilationSizes_;
        std::vector<int64_t> upsampleRates_;
        int64_t upsampleInitialChannel_;
        std::vector<int64_t> upsampleKernelSizes_;
        int64_t ginChannels_;
        int64_t numKernels_;
        int64_t numUpsamples_;
        modules::NormalizedConv1d convPre_{nullptr};
        std::vector<modules::NormalizedConvTranspose1d> ups_;
        std::vector<modules::ResBlock1> resblocks1_;
        std::vector<modules::ResBlock2> resblocks2_;
        modules::NormalizedConv1d convPost_{nullptr};
        torch::nn::Conv1d cond_{nullptr};
    };
    TORCH_MODULE(Generator);

    using DiscriminatorOutput = std::pair<torch::Tensor, std::vector<torch::Tensor>>;

    class DiscriminatorPImpl : public torch::nn::Module
    {
    public:
        DiscriminatorPImpl(
            int64_t period,
            int64_t kernelSize = 5,
            int64_t stride = 3,
            bool useSpectralNorm = false);

        DiscriminatorOutput forward(const torch::Tensor& x);

    private:
        int64_t period_;
        int64_t kernelSize_;
        int64_t stride_;
        bool useSpectralNorm_;
        std::vector<modules::NormalizedConv2d> convs_;
        modules::NormalizedConv2d convPost_{nullptr};
    };
    TORCH_MODULE(DiscriminatorP);

    class DiscriminatorSImpl : public torch::nn::Module
    {
    public:
        explicit DiscriminatorSImpl(bool useSpectralNorm = false);

        DiscriminatorOutput forward(const torch::Tensor& x);

    private:
        bool useSpectralNorm_;
        std::vector<modules::NormalizedConv1d> convs_;
        modules::NormalizedConv1d convPost_{nullptr};
    };
    TORCH_MODULE(DiscriminatorS);

    using MultiPeriodDiscriminatorOutput = std::tuple<
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<std::vector<torch::Tensor>>,
        std::vector<std::vector<torch::Tensor>>>;

    class MultiPeriodDiscriminatorImpl : public torch::nn::Module
    {
    public:
        explicit MultiPeriodDiscriminatorImpl(bool useSpectralNorm = false);

        MultiPeriodDiscriminatorOutput forward(
            const torch::Tensor& y,
            const torch::Tensor& yHat);

    private:
        bool useSpectralNorm_;
        DiscriminatorS discS_{nullptr};
        std::vector<DiscriminatorP> discPs_;
    };
    TORCH_MODULE(MultiPeriodDiscriminator);

    struct SynthesizerInferResult
    {
        torch::Tensor audio;
        torch::Tensor attention;
        torch::Tensor yMask;
        torch::Tensor z;
        torch::Tensor zP;
        torch::Tensor mP;
        torch::Tensor logsP;
    };

    struct SynthesizerForwardResult
    {
        torch::Tensor audio;
        torch::Tensor lengthLoss;
        torch::Tensor attention;
        torch::Tensor idsSlice;
        torch::Tensor xMask;
        torch::Tensor yMask;
        torch::Tensor z;
        torch::Tensor zP;
        torch::Tensor mP;
        torch::Tensor logsP;
        torch::Tensor mQ;
        torch::Tensor logsQ;
    };

    class SynthesizerTrnImpl : public torch::nn::Module
    {
    public:
        SynthesizerTrnImpl(
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
            int64_t nSpeakers = 0,
            int64_t ginChannels = 0,
            bool useSdp = true);

        SynthesizerForwardResult forward(
            const torch::Tensor& x,
            const torch::Tensor& xLengths,
            const torch::Tensor& y,
            const torch::Tensor& yLengths,
            const c10::optional<torch::Tensor>& sid = c10::nullopt);

        SynthesizerInferResult infer(
            const torch::Tensor& x,
            const torch::Tensor& xLengths,
            const c10::optional<torch::Tensor>& sid = c10::nullopt,
            double noiseScale = 1.0,
            double lengthScale = 1.0,
            double noiseScaleW = 1.0,
            c10::optional<int64_t> maxLength = c10::nullopt);

    private:
        int64_t nVocab_;
        int64_t specChannels_;
        int64_t segmentSize_;
        int64_t interChannels_;
        int64_t hiddenChannels_;
        int64_t filterChannels_;
        int64_t nHeads_;
        int64_t nLayers_;
        int64_t kernelSize_;
        double pDropout_;
        int64_t nSpeakers_;
        int64_t ginChannels_;
        bool useSdp_;
        TextEncoder encP_{nullptr};
        PosteriorEncoder encQ_{nullptr};
        Generator dec_{nullptr};
        ResidualCouplingBlock flow_{nullptr};
        StochasticDurationPredictor dpSdp_{nullptr};
        DurationPredictor dp_{nullptr};
        torch::nn::Embedding embG_{nullptr};
    };
    TORCH_MODULE(SynthesizerTrn);
}
